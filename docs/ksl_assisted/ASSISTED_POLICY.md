# Keypoint-first CNN assistance
Default mode: assisted. Confident keypoint predictions pass immediately without CNN.
Only uncertain keypoint predictions ask the CNN to corroborate the same candidate.
If CNN disagrees, is uncertain, or fails, output remains rejected.
CNN never produces an output without usable keypoints.
For corroborating borderline predictions only, scores use keypoint 80% and CNN 20%;
this is an initial engineering setting, not a measured webcam accuracy improvement.
Live landmark reuse and lazy CNN loading remain enabled.
Keys: 1 keypoint only, 2 RGB diagnostic, 3 assisted. Legacy fusion remains --mode fusion.
Existing score/margin thresholds (0.75/0.15) are retained; scores are not calibrated probabilities.
