from heapq import heappush, heappop, heapify
from math import radians, cos, sin, asin, sqrt
from typing import Tuple, Optional

class MinHeap:

    # Constructor to initialize a heap
    def __init__(self):
        self.heap = []

    def parent(self, i):
        return (i - 1) // 2

    # Inserts a new key 'k'
    def insertKey(self, k):
        heappush(self.heap, k)

        # Decrease value of key at index 'i' to new_val

    # It is assumed that new_val is smaller than heap[i]
    def decreaseKey(self, i, new_val):
        self.heap[i][0] = new_val
        # print(self.heap[self.parent(i)][0])
        while (i != 0 and self.heap[self.parent(i)][0] > self.heap[i][0]):
            # Swap heap[i] with heap[parent(i)]
            self.heap[i][0], self.heap[self.parent(i)][0] = self.heap[self.parent(i)][0], self.heap[i][0]
            self.heap[i][1], self.heap[self.parent(i)][1] = self.heap[self.parent(i)][1], self.heap[i][1]

    # Method to remove minium element from min heap
    def extractMin(self):
        return heappop(self.heap)

        # This functon deletes key at index i. It first reduces

    # value to minus infinite and then calls extractMin()
    def deleteKey(self, i):
        self.decreaseKey(i, float("-inf"))
        self.extractMin()

        # Get the minimum element from the heap

    def getMin(self):
        return self.heap[0]


def dijkstra(adj_list: dict, start_vertex: int, weights: Optional[dict] = None) -> Tuple[dict]:
    counter = 0
    curr_num = 0
    dist = {}
    preds = {}
    heap = MinHeap()
    if weights == None:
        weights = {}
        no_weights = True
    else:
        no_weights = False
    for key in adj_list.keys():
        if key == start_vertex:
            dist[key] = 0
            preds[key] = None
        else:
            dist[key] = float("inf")
            preds[key] = None
        if no_weights:
            weights[key] = 1
        heap.insertKey([dist[key], key])
    while len(heap.heap):
        distance, vertex_id = heap.extractMin()
        if distance > dist[vertex_id]:
            continue
        neighs = adj_list[vertex_id]
        for neigh in neighs.keys():
            edge_dist = adj_list[vertex_id][neigh][0]['length'] * weights[neigh]
            if edge_dist + distance < dist[neigh]:
                dist[neigh] = edge_dist + distance
                preds[neigh] = vertex_id
                heap.insertKey([edge_dist + distance, neigh])
    return (dist, preds)


def nearest_list_for_list(adj_list: dict, list1: list, list2: list, weights: Optional[dict] = None) -> dict:
    nearest = {}
    for obj in list1:
        distances, _ = dijkstra(adj_list, obj, weights)
        min_ = float('inf')
        min_id = -1
        for obj2 in list2:
            if distances[obj2] < min_:
                min_id = obj2
                min_ = distances[obj2]
        nearest[obj] = (min_, min_id)
    return nearest


def nearest_fwd_bwd_list_for_list(adj_list: dict, list1: list, list2: list, weights: Optional[dict] = None) -> dict:
    nearest = {}
    distances_fwd = {}
    distances_bwd = {}
    for obj in list1:
        distances_fwd[obj], _ = dijkstra(adj_list, obj, weights)
    for obj in list2:
        distances_bwd[obj], _ = dijkstra(adj_list, obj, weights)

    for obj in list1:
        min_ = float("inf")
        min_id = -1
        for obj2 in list2:
            if distances_fwd[obj][obj2] + distances_bwd[obj2][obj] < min_:
                min_ = distances_fwd[obj][obj2] + distances_bwd[obj2][obj]
                min_id = obj2
        nearest[obj] = (min_, min_id)
    return nearest


def nearest_bwd_list_for_list(adj_list: dict, list1: list, list2: list, weights: Optional[dict] = None) -> dict:
    nearest = {}
    distances_bwd = {}
    for obj in list2:
        distances_bwd[obj], _ = dijkstra(adj_list, obj, weights)

    for obj in list1:
        min_ = float("inf")
        min_id = -1
        for obj2 in list2:
            if distances_bwd[obj2][obj] < min_:
                min_ = distances_bwd[obj2][obj]
                min_id = obj2
        nearest[obj] = (min_, min_id)
    return nearest


def distances_fwd(adj_list: dict, list1: list, list2: list, weights: Optional[dict] = None) -> Tuple[dict]:
    distances = {}
    preds = {}
    for obj in list1:
        distances[obj], preds[obj] = dijkstra(adj_list, obj, weights)
    return (distances, preds)


def distances_bwd(adj_list: dict, list1: list, list2: list, weights: Optional[dict] = None) -> Tuple[dict]:
    distances = {}
    preds = {}
    for obj2 in list2:
        dist, pred = dijkstra(adj_list, obj2, weights)
        for obj in list1:
            if obj not in distances:
                distances[obj], preds[obj] = {obj2: dist[obj]}, pred
            else:
                distances[obj].update({obj2: dist[obj]})
    return (distances, preds)


def distances_fwd_bwd(adj_list: dict, list1: list, list2: list, weights: Optional[dict] = None) -> Tuple[dict]:
    distances = {}
    distances_fwd = {}
    distances_bwd = {}
    preds_fwd = {}
    preds_bwd = {}
    preds = {}
    for obj in list1:
        distances_fwd[obj], preds_fwd[obj] = dijkstra(adj_list, obj, weights)
    for obj in list2:
        distances_bwd[obj], preds_bwd[obj] = dijkstra(adj_list, obj, weights)

    for obj in list1:
        for obj2 in list2:
            if obj not in distances:
                distances[obj], preds[obj] = {obj2: distances_fwd[obj][obj2] + distances_bwd[obj2][obj]}, {
                    'fwd': preds_fwd[obj],
                    'bwd': preds_bwd[obj2]}
            else:
                distances[obj].update({obj2: distances_fwd[obj][obj2] + distances_bwd[obj2][obj]})
    return (distances, preds)