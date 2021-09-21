# AWARE
Adaptive Weighted voting AggRegation for Ensemble of classifiers.

Random Forest (RF) is a successful technique of ensemble prediction that uses the majority voting. 
However, it is clear that each tree in a random forest can have different contribution to the treatment of some instance.
So we need to replace the classical ordinary vote by the weighted one with local performance of each tree, this choice is justified by the fact that
the classical vote gives equal weight to each decision of each tree and depends on the choice of a majority of classifiers that give the same class for databases, 
while the trees do not have the same performance.

In this repo, by the inspiration of AdaBoost, I proposed a new way to calculate weights and named it AWARE. 
Experiments also indicate that this weighted voting method gives better results compared to the majority vote (RF) and 
all the other weighted voting methods (like TWRF, WAVE, DIRF).

See mean accuracy (%) results in the Table below.
| Dataset |  RF  | TWRF | WAVE | DIRF | AWARE |
|  ----   | ---- | ---- | ---- | ---- | ----  |
| breast | 95.4631 |	95.4368 |	95.4725 |	95.3852 | **95.6179** |
| car |	70.0239 |	70.0239 |	70.0436 |	70.0556 |	**78.2606** |
| credit |	70.0220 |	70.0260 |	70.1460 |	70.8000 |	**72.9940** |
| ecoli |	73.3853 |	75.1242 |	76.3744 |	75.3214 |	78.7455 |	**78.7988** |
| forest |	87.9065 |	87.9636 |	87.9657 |	87.8320 |	87.4520 |
| glass |	87.1992 |	92.3619 |	92.1038 |	92.5926 |	90.7088 |
| hcv |	90.4296 |	90.4476 |	90.4831 |	90.5297 |	**91.9214** |
| image |	92.7463 |	92.9190 |	93.1853 |	93.5986 |	**95.5368** |
| immuno |	78.6778 |	78.7333 |	78.7444 |	79.3043 |	**79.9667** |
| letter |	69.9882 |	70.2906 |	70.4752 |	70.8571 |	70.3827 |
| liver |	67.7068 |	67.7778 |	67.8953 |	67.3678 |	67.7829 |
| nursery |	58.9018 |	72.6107 |	78.8792 |	80.7143 |	**83.7187** |
| parkinsons |	87.8452 |	87.7895 |	88.0123 |	87.551 |	**89.5794** |
| shuttle |	99.7112 |	99.7227 |	99.7473 |	99.7342 |	**99.8384** |
| sonar |	80.1979 |	80.0850 |	80.2457 |	80.7036 |	**81.0457** |
| thoraric |	85.1064 |	85.1064 |	85.1064 |	86.4407 |	83.6915 |
| tic-tac-toe |	70.7223 |	70.8258 |	71.3973 |	78.7500 |	**83.0315** |
| transfusion |	77.1141	| 77.1369 |	77.1558 |	77.4606 |	75.4813 |
| waveform |	82.0916 |	82.1916 |	82.3948 |	82.6400 |	**83.3883** |
| wilt |	94.6069 |	94.6069 |	94.9126 |	94.6802 | **97.7844** |
