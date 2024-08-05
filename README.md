# SOTIF-Oriented Risk Assessment: A Multi-Dimensional Model for Autonomous Driving


## Abstract
Risk assessment is crucial for quantifying driving environment risks and reducing the Safety of the Intended Functionality (SOTIF) uncertainty in Autonomous Vehicles (AVs). Traditional methods, however, often concentrate on single-scenario metrics and are insufficient in capturing the complexities of multi-participant and multi-directional interactions. This paper introduces a risk assessment model capable of evaluating risks across diverse traffic participants, multiple directions, and complex multi-risk scenarios. Our model leverages state information from both the ego vehicle and surrounding traffic to compute a comprehensive risk probability distribution and environmental impact assessment. It achieves more precise risk quantification through an iterative process that calculates cumulative environmental costs and derives scalar risk values via sampling and summation within a map grid. Rigorously validated using extensive real-world datasets (inD, highD, rounD), our model demonstrates superior accuracy and scalability in diverse driving scenarios, as confirmed by comparative analysis with traditional metrics. Additionally, testing across various scenario types and comparing with existing single-scenario metrics highlighted the model's generalization capability. The results showed that our model not only accurately quantifies environmental risks but also significantly enhances the safety and reliability of AVs, especially in high-risk, long-tail cases, making it a robust tool for the development and deployment of safer autonomous driving systems.

This paper is currently being submitted to IEEE Robotics and Automation Letters (RA-L).
## Framework Overview

![Framework](figure/framework.png)
*Figure 1: Overview of our proposed risk assessment model.*

## Demo

### Example 1: Highway Scenario

![Framework](figure/highD_01_01.gif)|![Framework](figure/highD_01_769.gif)|![Framework](figure/highD_01_11003.gif)
---|---|---|

*Figure 2-1: risk model in highway scenario.*

### Example 2: Intersection Scenario
![Framework](figure/ind_00_7692.gif)|![Framework](figure/ind_00_10996.gif)|![Framework](figure/ind_00_4472.gif)
---|---|---|

*Figure 2-2: risk model in intersection scenario.*
### Example 3: Roundabout Scenario
![Framework](figure/round_00_501.gif)|![Framework](figure/round_00_1272.gif)|![Framework](figure/round_10_3821.gif)
---|---|---|

*Figure 2-3: risk model in roundabout scenario.*
## Quick Start

### Prerequisites

- Python 3.8 (Tested only in Python 3.8, other versions not verified)
- Required Python packages listed in `requirements.txt`

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/risk_assessment_model.git
    cd risk_assessment_model
    ```

2. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```
    
    OR 
    
    manual install packages: numpy,scipy,pandas,matplotlib and opencv-python
    ```bash
    pip install numpy scipy pandas matplotlib opencv-python
    
    ```
3. Prepare the datasets:
    - Download the required datasets from the following sources:
        - [inD Dataset](https://www.ind-dataset.com/)
        - [highD Dataset](https://www.highd-dataset.com/)
        - [rounD Dataset](https://www.round-dataset.com/)
    - Extract the datasets to a designated directory (e.g., `data/`):
        ```bash
        mkdir data
        unzip path_to_ind_dataset.zip -d data/
        unzip path_to_highd_dataset.zip -d data/
        unzip path_to_round_dataset.zip -d data/
        ```
    - Ensure the directory structure is as follows:
        ```
        risk_assessment_model/
        ├── data/
        │   ├── inD/
        │   ├── highD/
        │   └── rounD/
        └── ...
        ```

### Quick Start

1. Run the `quick_run.py` file:
    ```python
    python quick_run.py
    ```

## Datasets

Our model was validated using the following real-world datasets:

- [inD](https://levelxdata.com/ind-dataset/)
- [highD](https://levelxdata.com/highd-dataset/)
- [rounD](https://levelxdata.com/round-dataset/)

## Contributions

We welcome contributions to enhance this project. Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

