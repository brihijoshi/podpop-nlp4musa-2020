# Data

## Accessing raw transcripts

The labels and IDs of the podcasts are present in `popularity_train.txt` and `popularity_test.txt`. For instructions on downloading the raw transcripts and data descriptions, we refer you to [Longqi Yang's repository](https://github.com/ylongqi/podcast-data-modeling#data-descriptions).

We also acknowledge Yang and other authors for making their dataset publicly accessible without which this project would not have been possible.

## Accessing processed transcipts

Our processed version of the transripts can be accessed from [here](https://drive.google.com/drive/folders/0B5HoNU8jNviVfk83ejlRX2w3bkd2WkI0ZVBWS3djWDh4cGhweVh0bXZ3NE02NEJweWJnZ3c?resourcekey=0-LYal3iiqv1Sf5S1pq5rSiA&usp=sharing).

## Generating Podcast Triplets

You can create triplets used for training and testing our model using `src/get_triplets.py` and the processed dataset.
