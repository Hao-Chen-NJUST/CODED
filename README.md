# CODED: A Dataset for Positive Unlabeled Learning in Circular Workpiece Defect Detection

The CODED dataset is a circular workpiece defect detection dataset for positive unlabeled learning. It comprises 7,594 high-resolution circular images collected from *six* different production lines. The data details and some examples of CODED are shown in the following table and figure.
The CODED dataset can also be downloaded from this link: https://drive.google.com/file/d/1kj90dpOZxY0SiuWcv4qjX8eCHuvxXQIc/view?usp=sharing

## Data Description
### Data Detials
|Name     |Resolution |Train_Positive |Train_Unlabeled|Test_Positive|Test_Negative|Total    |#Defect types|
|---      |---        |---            |---            |---          |---          |---      |---          |
|0        |1824x1824  |2,100          |2,368          |17           |250          |4,735    |2            |
|1        |2464x2056  |80             |79             |15           |117          |291      |3            |
|2        |2464x2056  |500            |489            |10           |462          |1,461    |12           |
|3        |2464x2056  |40             |41             |10           |82           |173      |7            |
|4        |2464x2056  |150            |150            |13           |346          |659      |5            |
|5        |2464x2056  |40             |40             |16           |179          |275      |4            |
|All      |2065x1911  |2,910          |3,167          |81           |1,436        |7,594    |33           |

### Data examples


## Requirements
* Python 3.8
* Pytorch 1.10.1
* torchvision 0.11.2
* CUDA 11.1
* matplotlib 3.7.1
* numpy 1.24.2
* Pillow 9.5.0
* scikit_image 0.19.3
* scikit_learn 1.2.0
* tqdm 4.65.0
Please refer to `requiremets.txt` and `requiremets_conda.txt` for specific environment requirements.
