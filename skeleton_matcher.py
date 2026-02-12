from dataclasses import dataclass, field
from typing import Dict

from networkx.algorithms.bipartite import extendability

import numpy as np
import networkx as nx

@dataclass
class MatchData:
    skt_ids: list = field(default_factory=list)
    scores: list = field(default_factory=list)
    epilines: list = field(default_factory=list)
    
@dataclass
class SkeletonMatch:
    cam_ref: int
    skt_ref_id: int
    matches_by_cam: Dict[int, MatchData] = field(default_factory=dict)
    

class SkeletonMatcher:
    
    def __init__(self, fundamentals: dict, config: dict, projection_matrices: dict = None):
        self.fundamentals = fundamentals
        self.config = config
        self.projection_matrices = projection_matrices
        self.line_color_per_cam = {
            i: (int(np.random.randint(0, 255)), int(np.random.randint(0, 255)), int(np.random.randint(0, 255)))
            for i in range(config['num_cameras'])
        }
    
        self.kp_weights_arr = np.array([
            config['kp_weights'].get(i, 1.0) for i in range(config['n_keypoints'])
        ])
        
    def _dist_p_l_vectorized(self, points, lines):
        
        # Distancia = |ax + by + c| / sqrt(a^2 + b^2)
        # Numerador: dot product de linha e ponto (homogêneo)
        
        numerators = np.abs(lines[:, 0] * points[:, 0] + lines[:, 1] * points[:, 1] + lines[:, 2])
        denominators = np.sqrt(lines[:, 0]**2 + lines[:, 1]**2)
        
        # Evita divisão por zero
        denominators[denominators == 0] = np.inf
        return numerators / denominators


    def _sampson_error_vectorized(self, pts1_h, pts2_h, F):

        """
        pts1_h, pts2_h: (K, 3) homogêneos
        F: (3, 3)
        Retorna: (K,) erro Sampson por keypoint
        """

        # Fx1
        line_on_image_2 = (F @ pts1_h.T).T  # (K, 3)

        # F^T x2
        line_on_image_1 = (F.T @ pts2_h.T).T  # (K, 3)

        # x2^T F x1
        # produto linha a linha
        e = np.sum(pts2_h * (F @ pts1_h.T).T, axis=1)

        numerator = e ** 2

        denominator = (
            line_on_image_2[:, 0] ** 2 +
            line_on_image_2[:, 1] ** 2 +
            line_on_image_1[:, 0] ** 2 +
            line_on_image_1[:, 1] ** 2
        )

        # evita divisão por zero
        denominator[denominator < 1e-12] = np.inf

        return numerator / denominator

    def _calculate_skeleton_compatibility_vectorized(self, sk1, sk2, F_1_to_2, F_2_to_1):
    

        # Identifica quais keypoints são válidos (não zero)
        valid_mask = ~((np.all(sk1 == 0, axis=1)) | (np.all(sk2 == 0, axis=1)))
        
        valid_mask = ~((np.all(sk1 == 0, axis=1)) | (np.all(sk2 == 0, axis=1)))
        
        if not np.any(valid_mask):
            return 0.0, None

        # Filtra apenas pontos válidos para cálculo
        pts1 = sk1[valid_mask] # (K, 2)
        pts2 = sk2[valid_mask] # (K, 2)
        weights = self.kp_weights_arr[valid_mask]

        # Converte para coordenadas homogêneas (K, 3)
        ones = np.ones((pts1.shape[0], 1))
        pts1_h = np.hstack([pts1, ones])
        pts2_h = np.hstack([pts2, ones])

        # # 1. Epilines no frame 2 geradas pelos pontos do frame 1
        # # l = F @ p.  Shape: (3, 3) @ (K, 3).T -> (3, K) -> Transpose -> (K, 3)
        # lines_on_2 = (F_1_to_2 @ pts1_h.T).T
        
        # # 2. Epilines no frame 1 geradas pelos pontos do frame 2
        lines_on_1 = (F_2_to_1 @ pts2_h.T).T
        
        # # Calcula distâncias vetorizadas
        # dists_on_2 = self._dist_p_l_vectorized(pts2, lines_on_2)
        # dists_on_1 = self._dist_p_l_vectorized(pts1, lines_on_1)
        
        sampson_errors = self._sampson_error_vectorized(
                                    pts1_h,
                                    pts2_h,
                                    F_1_to_2
                                )
        

        # print(sampson_errors)
        # exit()

        # Lógica de Threshold
        # max_dist = self.config['max_epipolar_dist']
        max_sampson = self.config.get('max_sampson_error', 5.0)

        
        # Pontos que satisfazem a condição em AMBOS os sentidos
        # match_mask = (dists_on_2 < max_dist) & (dists_on_1 < max_dist)
        match_mask = sampson_errors < max_sampson

        # Soma os pesos onde match_mask é True
        score = np.sum(weights[match_mask])
        
        full_lines_on_1 = np.full((self.config['n_keypoints'], 3), np.nan)
        full_lines_on_1[valid_mask] = lines_on_1
        
        return score, full_lines_on_1
    
    def _check_cycle_consistency(self, cam_a: int, skt_a_id: int, 
                                  cam_b: int, skt_b_id: int,
                                  cam_c: int, skt_c_id: int,
                                  skeletons_by_cam: list) -> float:
        """
        Verifica a consistência do ciclo A -> B -> C -> A
        
        Args:
            cam_a, skt_a_id: Câmera e ID do skeleton A
            cam_b, skt_b_id: Câmera e ID do skeleton B  
            cam_c, skt_c_id: Câmera e ID do skeleton C
            skeletons_by_cam: Lista de skeletons por câmera
            
        Returns:
            float: Score de consistência do ciclo (maior = mais consistente)
        """
        # Pega os skeletons
        sk_a = skeletons_by_cam[cam_a][skt_a_id]
        sk_b = skeletons_by_cam[cam_b][skt_b_id]
        sk_c = skeletons_by_cam[cam_c][skt_c_id]
        
        # Calcula compatibilidade A -> B
        F_a_to_b = self.fundamentals[cam_a][cam_b]
        F_b_to_a = self.fundamentals[cam_b][cam_a]
        score_ab, _ = self._calculate_skeleton_compatibility_vectorized(
            sk_a, sk_b, F_a_to_b, F_b_to_a
        )
        
        # Calcula compatibilidade B -> C
        F_b_to_c = self.fundamentals[cam_b][cam_c]
        F_c_to_b = self.fundamentals[cam_c][cam_b]
        score_bc, _ = self._calculate_skeleton_compatibility_vectorized(
            sk_b, sk_c, F_b_to_c, F_c_to_b
        )
        
        # Calcula compatibilidade C -> A (fecha o ciclo)
        F_c_to_a = self.fundamentals[cam_c][cam_a]
        F_a_to_c = self.fundamentals[cam_a][cam_c]
        score_ca, _ = self._calculate_skeleton_compatibility_vectorized(
            sk_c, sk_a, F_c_to_a, F_a_to_c
        )
        
        # Score do ciclo é a média geométrica (penaliza se algum link for fraco)
        if score_ab > 0 and score_bc > 0 and score_ca > 0:
            cycle_score = (score_ab * score_bc * score_ca) ** (1/3)
        else:
            cycle_score = 0.0
            
        return cycle_score

    def _validate_with_cycle_consistency(self, matches: Dict, 
                                         skeletons_by_cam: list,
                                         ids_by_cam: list) -> Dict:
        """
        Valida e filtra matches usando cycle consistency.
        
        Para cada par de matches (cam_a, sk_a) <-> (cam_b, sk_b),
        verifica se existe uma terceira câmera cam_c onde o ciclo
        A -> B -> C -> A é consistente.
        
        Args:
            matches: Dicionário de matches do método match()
            skeletons_by_cam: Lista de skeletons por câmera
            ids_by_cam: Lista de IDs por câmera
            
        Returns:
            Dict: Matches filtrados e com scores de cycle consistency
        """
        weight_cycle = self.config.get('weight_cycle', 0.5)
        
        validated_matches = {}
        
        for cam_ref, skeletons_data in matches.items():
            validated_matches[cam_ref] = {}
            
            for skt_ref_id, skt_match in skeletons_data.items():
                # Encontra o índice do skeleton na lista
                idx_ref = ids_by_cam[cam_ref].index(skt_ref_id)
                
                # Para cada câmera com matches
                for cam_other in skt_match.matches_by_cam.keys():
                    match_data = skt_match.matches_by_cam[cam_other]
                    
                    # Para cada skeleton candidato na outra câmera
                    validated_skts = []
                    validated_scores = []
                    validated_epilines = []
                    
                    for i, skt_other_id in enumerate(match_data.skt_ids):
                        original_score = match_data.scores[i]
                        
                        # Busca uma terceira câmera para fechar o ciclo
                        best_cycle_score = 0.0
                        
                        for cam_third in range(self.config['num_cameras']):
                            # Pula se for uma das câmeras já envolvidas
                            if cam_third == cam_ref or cam_third == cam_other:
                                continue
                            
                            # Pula se não houver skeletons nesta câmera
                            if len(skeletons_by_cam[cam_third]) == 0:
                                continue
                                
                            # Tenta cada skeleton da terceira câmera
                            for idx_third, skt_third in enumerate(skeletons_by_cam[cam_third]):
                                skt_third_id = ids_by_cam[cam_third][idx_third]
                                
                                # Verifica cycle consistency
                                cycle_score = self._check_cycle_consistency(
                                    cam_ref, idx_ref,
                                    cam_other, ids_by_cam[cam_other].index(skt_other_id),
                                    cam_third, idx_third,
                                    skeletons_by_cam
                                )
                                
                                best_cycle_score = max(best_cycle_score, cycle_score)
                        
                        # Combina score original com cycle consistency
                        combined_score = (1 - weight_cycle) * original_score + weight_cycle * best_cycle_score
                        
                        # Só adiciona se passar um threshold mínimo
                        min_cycle_threshold = self.config.get('min_cycle_score', 0.0)
                        if best_cycle_score >= min_cycle_threshold:
                            validated_skts.append(skt_other_id)
                            validated_scores.append(combined_score)
                            validated_epilines.append(match_data.epilines[i])
                    
                    # Atualiza os matches com os validados
                    if validated_skts:
                        if skt_ref_id not in validated_matches[cam_ref]:
                            validated_matches[cam_ref][skt_ref_id] = SkeletonMatch(
                                cam_ref=cam_ref,
                                skt_ref_id=skt_ref_id
                            )
                        
                        validated_matches[cam_ref][skt_ref_id].matches_by_cam[cam_other] = MatchData(
                            skt_ids=validated_skts,
                            scores=validated_scores,
                            epilines=validated_epilines
                        )
        
        return validated_matches
    
    def extract_skeletons_from_annotations(self, annotations):
        skeletons_by_cam = []
        ids_by_cam = []
            
        for i in range(self.config['num_cameras']):
            skeletons_for_current_cam = []
            ids_for_current_cam = []
            
            if i < len(annotations):
                for obj in annotations[i].objects:
                    skeleton = np.zeros((self.config['n_keypoints'], 2), dtype=np.float32)
                    for kp in obj.keypoints:
                        if kp.id < self.config['n_keypoints']:

                             skeleton[kp.id - 1] = [kp.position.x, kp.position.y]
                            
                    skeletons_for_current_cam.append(skeleton)
                    ids_for_current_cam.append(obj.id)
                    
            skeletons_by_cam.append(skeletons_for_current_cam)
            ids_by_cam.append(ids_for_current_cam)
        
        return skeletons_by_cam, ids_by_cam
        
    def organize_epilines_by_keypoint(self, skt_match: SkeletonMatch) -> Dict[int, np.ndarray]:
        lines_per_kp = {}
        for cam_other, match_data in skt_match.matches_by_cam.items():
                    for idx, lines in enumerate(match_data.epilines):
                        for kp_idx in range(self.config['n_keypoints']):
                            line = lines[kp_idx]
                            if not np.isnan(line).any():
                                lines_per_kp.setdefault(kp_idx, []).append((cam_other, match_data.skt_ids[idx], match_data.scores[idx], line))   
        return lines_per_kp
    
    def match(self, skeletons_by_cam: list, ids_by_cam: list, frames, use_cycle_consistency=True): 
    
        matches: Dict[int, Dict[int, SkeletonMatch]] = {}

        for cam_ref, F_ref_to_others in self.fundamentals.items():
            matches.setdefault(cam_ref, {})
            
            ids_ref_cam = ids_by_cam[cam_ref]
            skeletons_ref_cam = skeletons_by_cam[cam_ref]
            
            for idx_skt_ref, skt_ref in enumerate(skeletons_ref_cam):
            
                skt_ref_id = ids_ref_cam[idx_skt_ref]
            
                skt_match = SkeletonMatch(
                    cam_ref=cam_ref,
                    skt_ref_id=skt_ref_id,
                )
                
                for cam_other, F_ref_to_other in F_ref_to_others.items():
                    
                    skt_match.matches_by_cam[cam_other] = MatchData()
                    
                    skeletons_other_cam = skeletons_by_cam[cam_other]
                    ids_other_cam = ids_by_cam[cam_other]
                    
                    F_other_to_ref = self.fundamentals[cam_other][cam_ref]
                    
                    for idx_skt_other, skt_other in enumerate(skeletons_other_cam):
                        
                        skt_other_id = ids_other_cam[idx_skt_other]
                        
                        compatibility_score, full_lines_on_ref = self._calculate_skeleton_compatibility_vectorized(
                            skt_ref, skt_other, F_ref_to_other, F_other_to_ref
                        )
                        
                        min_matching_joints = self.config.get('min_matching_joints', 5)
                        
                        if compatibility_score >= min_matching_joints:
                            skt_match.matches_by_cam[cam_other].skt_ids.append(skt_other_id)
                            skt_match.matches_by_cam[cam_other].scores.append(compatibility_score)
                            skt_match.matches_by_cam[cam_other].epilines.append(full_lines_on_ref)
                
                matches[cam_ref][skt_ref_id] = skt_match
        
        if use_cycle_consistency:
            print("Aplicando validação por cycle consistency...")
            matches = self._validate_with_cycle_consistency(matches, skeletons_by_cam, ids_by_cam)
        
        refined_matches = self.refine_matches(matches)
        matched_persons = self.build_skeleton_groups(refined_matches)
        
        return matched_persons
    
    def refine_matches(self, matches: Dict[int, Dict[int, SkeletonMatch]]):
        
        refined_by_cam = {}
        
        for cam_ref, skeletons in matches.items():
            refined_by_cam[cam_ref] = {}
            
            for skt_ref_id, skt_match in skeletons.items():
                
                refined_by_cam[cam_ref][skt_ref_id] = {}
                
                lines_per_kp = self.organize_epilines_by_keypoint(skt_match)
                
                for kp_idx in range(self.config['n_keypoints']):
                    if kp_idx not in lines_per_kp:
                        refined_by_cam[cam_ref][skt_ref_id][kp_idx] = None
                        continue
                    
                    data = lines_per_kp[kp_idx]
                    if len(data) < 2:
                        refined_by_cam[cam_ref][skt_ref_id][kp_idx] = None
                        continue
                    
                    from itertools import combinations
                    
                    candidate_combinations = list(combinations(data, 2))
                    graph_dists = {}
                    graph_scores = {}
                    
                    for combo in candidate_combinations:
                        (cam_1, skt_1, score_1, line_1), (cam_2, skt_2, score_2, line_2) = combo
                        
                        point_intersect = self._compute_line_intersection_2d(line_1, line_2)
                        
                        if point_intersect is None:
                            continue
                        
                        dists_to_lines = []
                        for _, _, _, line_k in data:
                            dist = self._dist_p_l_vectorized(
                                np.array([point_intersect]), 
                                np.array([line_k])
                            )[0]
                            dists_to_lines.append(dist)
                        
                        avg_dist = np.mean(dists_to_lines)
                        
                        max_dist_threshold = self.config.get('max_intersection_dist', 8)
                        if avg_dist > max_dist_threshold:
                            continue
                        
                        avg_score = (score_1 + score_2) / 2.0
                        
                        key = (
                            (cam_1, skt_1),
                            (cam_2, skt_2)
                        )
                        
                        graph_dists[key] = avg_dist
                        graph_scores[key] = avg_score
                    
                    if len(graph_scores) == 0:
                        refined_by_cam[cam_ref][skt_ref_id][kp_idx] = None
                        continue
                    
                    best_combination = self._select_best_combination(graph_scores, graph_dists)
                    best_combination['avg_dist'] = graph_dists[best_combination['original_key']]
                    
                    refined_by_cam[cam_ref][skt_ref_id][kp_idx] = best_combination
                    
        return refined_by_cam
        
    def _select_best_combination(self, graph_scores, graph_dists):
        
        scores = np.array(list(graph_scores.values()))
        dists = np.array([graph_dists[k] for k in graph_scores.keys()])
        keys = list(graph_scores.keys())
        
        # Normalização
        score_range = scores.max() - scores.min()
        dist_range = dists.max() - dists.min()
        
        if score_range > 1e-8:
            score_norm = (scores - scores.min()) / score_range
        else:
            # Se todos os scores são iguais, normaliza para 0.5
            score_norm = np.full_like(scores, 0.5)
        
        if dist_range > 1e-8:
            dist_norm = (dists - dists.min()) / dist_range
        else:
            # Se todas as distâncias são iguais, normaliza para 0.0 (melhor caso)
            dist_norm = np.zeros_like(dists)
        
        weight_dist = self.config.get('weight_distance')  
        weight_score = self.config.get('weight_score')   
        
        total = weight_dist + weight_score
        
        if total > 0: # se total < 0 weight_dist = self.config.get('weight_distance') e weight_score = self.config.get('weight_score')  
            weight_dist /= total
            weight_score /= total
        
        # Custo combinado (menor é melhor)
        # +dist_norm para MINIMIZAR distância (valores pequenos = bom)
        # -score_norm para MAXIMIZAR score (valores grandes = bom)
        combined_cost = weight_dist * dist_norm - weight_score * score_norm
        
        # Seleciona a combinação com MENOR custo
        best_idx = np.argmin(combined_cost)
        best_key = keys[best_idx]
        
        return {
            'combination': dict(best_key),
            'original_key': best_key,
            'score': graph_scores[best_key],
            'combined_cost': combined_cost[best_idx]
        }
            
    def build_skeleton_groups(self, refined_matches):
       
        G = nx.Graph()
        edge_weights = {}  #rastrear métricas das arestas
        
        for cam_ref, skeletons_data in refined_matches.items():
            for skt_ref_id, keypoints_data in skeletons_data.items():
                
                ref_node = (cam_ref, skt_ref_id)
                G.add_node(ref_node)
                
                connections_count = {}
                connections_costs = {}
                connections_dists = {}
                connections_scores = {}
                
                for kp_idx, best_combination in keypoints_data.items():
                    if best_combination is None:
                        continue
                    
                    for (cam_other, skt_other_id) in best_combination['combination'].items():
                        key = (cam_other, skt_other_id)
                        connections_count[key] = connections_count.get(key, 0) + 1
                        
                        # Acumula CUSTO COMBINADO (menor é melhor)
                        combined_cost = best_combination.get('combined_cost', 0)
                        if key not in connections_costs:
                            connections_costs[key] = 0
                        connections_costs[key] += combined_cost
                        
                        # Acumula DISTÂNCIA para análise
                        avg_dist = best_combination.get('avg_dist', 0)
                        if key not in connections_dists:
                            connections_dists[key] = 0
                        connections_dists[key] += avg_dist
                        
                        # Acumula score
                        if key not in connections_scores:
                            connections_scores[key] = 0
                        connections_scores[key] += best_combination.get('score', 1)
                        
                min_keypoints = self.config.get('min_keypoints_for_grouping', 10)
                
                for (cam_other, skt_other_id), count in connections_count.items():
                    
                    if count >= min_keypoints:
                        other_node = (cam_other, skt_other_id)
                        edge = tuple(sorted([ref_node, other_node]))
                        
                        # Calcula métricas médias
                        avg_cost = connections_costs[(cam_other, skt_other_id)] / count
                        avg_dist = connections_dists[(cam_other, skt_other_id)] / count
                        avg_score = connections_scores[(cam_other, skt_other_id)] / count
                        
                        # Custo menor = peso maior = conexão mais forte
                        # Inverte o custo para que menor custo = maior peso
                        edge_weight = 1.0 / (abs(avg_cost) + 1e-6)
                        
                        # Adiciona a aresta com o peso baseado no custo
                        G.add_edge(ref_node, other_node, 
                                  weight=edge_weight, 
                                  count=count,        
                                  avg_cost=avg_cost,  
                                  avg_dist=avg_dist,  
                                  avg_score=avg_score)
                        
                        edge_weights[edge] = {
                            'weight': edge_weight,
                            'count': count,
                            'avg_cost': avg_cost,
                            'avg_dist': avg_dist,
                            'avg_score': avg_score
                        }
        
        # Encontra componentes conectados
        matched_persons = []
        
        for component in nx.connected_components(G):
            # Resolve conflitos: mesma câmera com múltiplos esqueletos
            nodes_by_camera = {}
            for cam_id, skt_id in component:
                if cam_id not in nodes_by_camera:
                    nodes_by_camera[cam_id] = []
                nodes_by_camera[cam_id].append((cam_id, skt_id))
            
            # Para cada câmera, escolhe o esqueleto com mais conexões fortes
            group_ids = {}
            for cam_id, nodes in nodes_by_camera.items():
                if len(nodes) == 1:
                    group_ids[cam_id] = nodes[0][1]
                else:
                    # Múltiplos esqueletos da mesma câmera - escolhe com MAIOR peso total (que corresponde ao MENOR custo médio)
                    best_node = max(nodes, key=lambda n: G.degree(n, weight='weight'))
                    group_ids[cam_id] = best_node[1]
            
            # Só aceita grupos com pelo menos 2 câmeras
            if len(group_ids) < 2:
                continue
            
            # Calcular métricas médias do grupo baseado nas arestas
            total_cost = 0
            total_dist = 0
            total_score = 0
            total_edges = 0
            subgraph = G.subgraph(component)
            
            for (n1, n2) in subgraph.edges():
                edge = tuple(sorted([n1, n2]))
                if edge in edge_weights:
                    total_cost += edge_weights[edge]['avg_cost']
                    total_dist += edge_weights[edge]['avg_dist']
                    total_score += edge_weights[edge]['avg_score']
                    total_edges += 1
            
            avg_group_cost = total_cost / total_edges if total_edges > 0 else 0
            avg_group_dist = total_dist / total_edges if total_edges > 0 else 0
            avg_group_score = total_score / total_edges if total_edges > 0 else 0
            
            matched_persons.append({
                'ids': group_ids,             
                'score': avg_group_score,     
                'avg_cost': avg_group_cost,   
                'avg_dist': avg_group_dist,   
                'num_cameras': len(group_ids),
                'num_edges': total_edges,     
            })
        
        # Prioriza grupos com:
        # - Mais câmeras
        # - Menor custo médio
        matched_persons.sort(
            key=lambda x: (
                x['num_cameras'],
                -x['avg_cost']   
            ), 
            reverse=True
        )
        
        return matched_persons            
                        
    def _compute_line_intersection_2d(self, line1, line2):
        
        a1, b1, c1 = line1
        a2, b2, c2 = line2
        
        det = a1 * b2 - a2 * b1
        
        if abs(det) < 1e-6:  # Linhas paralelas
            return None
        
        x = (b1 * c2 - b2 * c1) / det
        y = (a2 * c1 - a1 * c2) / det
        
        return np.array([x, y])