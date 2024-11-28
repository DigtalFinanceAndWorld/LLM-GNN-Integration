![image](https://github.com/user-attachments/assets/b296fb31-1794-451b-8980-b37d212b5cf3)# LLM-GNN: An Integrated Framework for Enhanced Time-Space Modeling and Predictive Decision-Making

How can Large Language Models (LLMs) be integrated with Graph Neural Networks (GNNs) to enhance the effectiveness of joint time-space (graph) modeling and predictive decision-making? This paper proposes an innovative integrated framework named LLM-GNN, which combines the advantages of LLMs and GNNs to improve the capability of joint modeling in complex scenarios involving temporal evolution and spatial diffusion. As a practical application example, this study implements the integrated framework to tackle the Ethereum phishing fraud detection problem. By leveraging the LLM for deep semantic analysis from a multi-resolution perspective on transaction information and network structure, providing insights distinct from those of GNNs, we capture special structural features and extract potential risk signals. Meanwhile, GNNs concentrate on identifying abnormal patterns within the transaction network structure. Subsequently, these two types of risk signals are organically synthesized to form a comprehensive risk assessment system.

### Dataset Download:
[https://drive.google.com/drive/folders/18MNqjRBPRJWBX_Met3XW81TlHp47lLv1?usp=sharing](https://drive.google.com/drive/folders/18MNqjRBPRJWBX_Met3XW81TlHp47lLv1?usp=sharing)

## Our innovations:

1. A novel integration framework incorporating the risk signals extracted by LLMs and GNNs, leveraging their complementary strengths to enhance phishing detection;
2. An innovative Multi-Resolution Analysis & Forecasting paradigm for unlocking LLMs' huge potential in extracting distinct risk signals while significantly reducing inference costs;
3. An innovative strategy for representative train sample selection;
4. Three highly effective fine-tuning strategies (BC-FT, SRA-FT, CRRA-FT) that incrementally introduce risk features and Chain-of-Thought (CoT) reasoning mechanisms.

## Our empirical testing:

To evaluate our framework's real-world effectiveness, we constructed a "high-fidelity" dynamic rolling window train-test platform that closely mimics real-world scenarios under a temporal evolution perspective. 

1.	Real-World Scenario Alignment: Our dynamic data par- titioning incorporates a label delay ∆Tlabel to simulate the time required by regulatory agencies to analyze transac- tions and confirm phishing activity. Specifically, labels for transactions at time Ti become available only after a delay of ∆Tlabel. This ensures that the training set only utilizes historical data T < Ti + ∆Tlabel, preventing data leakage and aligning the evaluation process with real-world de- tection workflows. Previous studies overlook the impact of label delay, our framework uniquely accounts for this critical factor. By explicitly considering label delay, the proposed approach ensures a more realistic and practical evaluation of phishing detection models.
2.	Dynamic Rolling Window Train-Test Framework: We developed a forward-rolling window train-test framework to reflect the continuous nature of real-world transaction monitoring. Models are trained on historical data and tested on unseen future data, effectively capturing the temporal dynamics of transaction networks and adapting to evolving node labels. This framework not only aligns with real-world operational workflows but also serves as a critical step toward real-time phishing detection, offer- ing scalability and adaptability for practical applications in dynamic monitoring systems.
3.	Comprehensive Evaluation Metrics: We employed TPR@10%FPR to measure the model’s ability to detect true positives under strict false alarm constraints, essential for phishing detection in high-stakes scenarios. The standard deviation of AUC assesses performance stability, crucial for dynamic transaction networks. The Sharpe Ratio of AUC, inspired by financial risk analysis, quantifies the balance between high AUC and low variability, ensuring reliable and robust decision-making in uncertain environments.

Empirical tests demonstrate that the proposed LLM-GNN framework achieved an AUC of 91.38% (25% increase than ZipZap) with TPR@10%FPR of 72.42% (a 95% jump compared to ZipZap), significantly surpassing traditional methods in performance. It is particularly noteworthy that our framework significantly improves the shape of the ROC curves, elevating its performance to the level that is truly applicable in real-world scenarios. Moreover, our framework is highly generalizable and can be applied to a broader range of domains and applications aiming at enhancing the effectiveness of joint time-space modeling and predictive decision-making.


## Potential use-cases:

1. Enhancing forecasting performance: Our LLM-GNN framework demonstrates superior predictive capabilities compared to traditional methods.
2. Enhancing ROC curve shape and TPR@10%FPR: The framework significantly enhances the ROC curve shape, particularly improving the True Positive Rate at 10% False Positive Rate (TPR@10%FPR). This improvement elevates the model's performance to a level that achieves real-world usability, addressing a critical requirement for practical application.
3. Enhancing performance in highly imbalanced sample scenarios: The LLM-GNN approach shows robust performance even in situations where the distribution of positive and negative samples is highly skewed, a common challenge in real-world fraud detection tasks.


