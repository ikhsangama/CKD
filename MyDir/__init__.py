#APPLYING THE KNN METHOD

# import matplotlib.pyplot as plt
import scipy.stats as ss
import numpy as np
import random

#2 FINDING DISTANCE BETWEEN TWO POINTS
def distance(p2,p1):
    distance = (np.sqrt(np.sum((np.power(p2-p1,2)))))
    return distance

#3 MAJORITY VOTE
def majority_vote(votes):
    vote_counts = {}
    for vote in votes:
        if vote in vote_counts:
            vote_counts[vote] += 1
        else:
            vote_counts[vote] = 1
    winners = []
    max_count = max(vote_counts.values())
    # max_counts
    for vote, count in vote_counts.items():
        if count == max_count:
            winners.append(vote)
    return random.choice(winners)

#4 FINDING NEAREST NEIGHBOURS
def find_nearest_neightbors(p,points,k=5):
    """mencari k terdekat dari titik p dan mengembalikan isi indeksnya"""
    #loop over all points
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
    #compute the distance between point p and every other point
        distances[i] = distance(p,points[i])
    #sort distance and return those k points that are nearest to point p
    ind = np.argsort(distances)
    return ind[:k]

def knn_predict (p,points, outcomes, k=5):
    #menemukan k titik terdekat
    ind = find_nearest_neightbors(p,points,k)
    return majority_vote(outcomes[ind])