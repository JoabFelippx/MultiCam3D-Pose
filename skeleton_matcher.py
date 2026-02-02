from dataclasses import dataclass, field
from typing import Dict

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
    
    def __init__(self, fundamentals: dict, config: dict):
        self.fundamentals = fundamentals
        self.config = config
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

        # 1. Epilines no frame 2 geradas pelos pontos do frame 1
        # l = F @ p.  Shape: (3, 3) @ (K, 3).T -> (3, K) -> Transpose -> (K, 3)
        lines_on_2 = (F_1_to_2 @ pts1_h.T).T
        
        # 2. Epilines no frame 1 geradas pelos pontos do frame 2
        lines_on_1 = (F_2_to_1 @ pts2_h.T).T
        
        # Calcula distâncias vetorizadas
        dists_on_2 = self._dist_p_l_vectorized(pts2, lines_on_2)
        dists_on_1 = self._dist_p_l_vectorized(pts1, lines_on_1)
        
        # Lógica de Threshold
        max_dist = self.config['max_epipolar_dist']
        
        # Pontos que satisfazem a condição em AMBOS os sentidos
        match_mask = (dists_on_2 < max_dist) & (dists_on_1 < max_dist)
        
        # Soma os pesos onde match_mask é True
        score = np.sum(weights[match_mask])
        
        full_lines_on_1 = np.full((self.config['n_keypoints'], 3), np.nan)
        full_lines_on_1[valid_mask] = lines_on_1
        
        return score, full_lines_on_1
    
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
    
    def match(self, skeletons_by_cam: list, ids_by_cam: list, frames,): 
    
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
                    
                    for idx_skt_other, skt_other in enumerate(skeletons_other_cam):
                        
                        skt_other_id = ids_other_cam[idx_skt_other]

                        score, lines_on_1 = self._calculate_skeleton_compatibility_vectorized(
                            skt_ref, 
                            skt_other, 
                            F_ref_to_other,
                            self.fundamentals[cam_other][cam_ref],
                        )
                        
                        if lines_on_1 is None:
                            continue
                        
                        if score >= self.config['min_matching_joints']:
                            entry = skt_match.matches_by_cam[cam_other]
                            entry.skt_ids.append(skt_other_id)
                            entry.scores.append(score)
                            entry.epilines.append(lines_on_1)
                        
                matches[cam_ref][skt_ref_id] = skt_match
        
        refined_matches = self.filter_intersection_of_epipolar_lines(skeletons_by_cam, ids_by_cam,matches, frames)
        matched_persons = self.build_skeleton_groups(refined_matches)
        
        return matched_persons

    def filter_intersection_of_epipolar_lines(self, skeletons_by_cam, ids_by_cam, matches: Dict[int, Dict[int, SkeletonMatch]], frames):
        
        refined_matches = {}
        for cam_ref, skt_matches_by_id in matches.items():
            
            refined_matches.setdefault(cam_ref, {})
            current_ids_list = ids_by_cam[cam_ref]
            
            for skt_ref_id, skt_match in skt_matches_by_id.items():
                if skt_ref_id not in current_ids_list:
                    continue
            
                ref_index = current_ids_list.index(skt_ref_id)
                skeleton_ref = skeletons_by_cam[cam_ref][ref_index]
                
                lines_per_kp = self.organize_epilines_by_keypoint(skt_match)
                
                # para cada keypoint, encontrar a melhor correspondência
                best_matches_for_skeleton = {}
                
                for kp_idx, lines_info in lines_per_kp.items():
                
                    kp_ref = skeleton_ref[kp_idx]
                    
                    # pula keypoints inválidos
                    if np.all(kp_ref == 0):
                        continue
                
                    # agrupar por camera as linhas epipolares
                    lines_by_cam = {}
                    for cam_other, skt_other_id, score, line in lines_info:
                        lines_by_cam.setdefault(cam_other, []).append(
                        {
                            'skt_id': skt_other_id,
                            'score': score,
                            'line': line
                        })
                        
                    best_combination = self._find_best_combination_for_keypoint(
                        kp_ref, lines_by_cam, cam_ref, frames)                    
                
                    if best_combination:
                        best_matches_for_skeleton[kp_idx] = best_combination
                        
                refined_matches[cam_ref][skt_ref_id] = best_matches_for_skeleton
        return refined_matches
    
    def _find_best_combination_for_keypoint(self, kp_ref, lines_by_cam, cam_ref, frames):
        
        cam_ids = list(lines_by_cam.keys())
        
        if len(cam_ids) < 2:
            return None
        
        graph_scores = {} # (skt_id_cam1, skt_id_cam2, ...)
        graph_dists = {}  # (skt_id_cam1, skt_id_cam2, ...)
        
        for i in range(len(cam_ids)):
            for j in range(len(cam_ids)):
                if i == j:
                    continue
                
                cam_i = cam_ids[i]
                cam_j = cam_ids[j]
                
                lines_info_i = lines_by_cam[cam_i]
                lines_info_j = lines_by_cam[cam_j]
                
                for line_data_i in lines_info_i:
                    for line_data_j in lines_info_j:
                        
                        intersection_pt = self._compute_line_intersection_2d(
                            line_data_i['line'], line_data_j['line'])
                        
                        if intersection_pt is None:
                            continue
                            
                        dist = np.linalg.norm(intersection_pt - kp_ref)    
                        
                        if dist <= self.config.get('max_intersection_dist', 7):
                            key = tuple(
                                sorted([
                                    (cam_i, line_data_i['skt_id']),
                                    (cam_j, line_data_j['skt_id'])
                                ])
                            )
                                
                            score = line_data_i['score'] + line_data_j['score']
                            
                            graph_scores[key] = score + graph_scores.get(key, 0)
                            graph_dists[key] = graph_dists.get(key, 0) + dist
                            
        if not graph_scores:
            return None
        
        # Normalizar scores e distâncias
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