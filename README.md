Limits of Laboratory Data for Early Colorectal Cancer Risk Detection
Hussein Al Zuhairi
alzuhairihussein@outlook.com


ABSTRACT
Finding colorectal cancer (CRC) early is important for screening but it’s difficult when relying mostly on lab results and symptoms. The goal of this study is to figure out whether it is possible to determine the early risk of CRC based on longitudinal electronic health records (EHR) while maintaining a low rate of false positives. Synthea-generated EHR data and machine learning was used for both development and training of the models included in this research. The Synthea EHR data had laboratory test results, patient symptoms, encounter data, patient and family medical histories, screening adherence and longitudinal risk data. The study showed that the lab-only models could not obtain high recall when faced with a tight false positive threshold. On the other hand, including additional risk-related data, not just laboratory data, provided an earlier and more accurate prediction. The model achieved a recall of 76.8%, and false positive rate of 7.1% during the training process; with a risk detection time of 12 months before a CRC diagnosis. This shows that the ability to detect early CRC risk will require much more information than simply a few laboratory tests.

INTRODUCTION
Identifying CRC early is important for screening and prevention, but it’s been proven to be difficult when using laboratory results and symptoms alone [1]. Many lab changes and gastrointestinal symptoms aren’t specific and often appear late which discourages their usefulness for early detection [2]. Also, screening tools must keep false-positives low since unnecessary follow-up testing could cause more unintended harm and heavily increase healthcare costs [5]. For this paper, the models considered a false-positive rate of around 10% to be ideal since anything lower could be resource-heavy and  just generating CRC patients might introduce leakages with the models. This does not reflect real-world metrics as doctors would consider this as a high false-positive rate.
EHRs contain longitudinal clinical information that gives more information than laboratory values [3]. This includes patient risk factors and patterns of healthcare use such as: a family history of CRC, following screening recommendations, and diagnostic testing over time. These factors might show a higher risk before diseases actually develop.
This paper examines whether or not early CRC risk can be found using longitudinal EHR data while maintaining low false-positive rates. Machine learning models were trained on using synthetic EHR data which includes: laboratory values, gastrointestinal symptoms, encounter patterns, and longitudinal risk information. Later, this paper compares using only laboratory data and models that include longitudinal risk context to find what information is reliable for early CRC risk detection. 
BACKGROUND
Although colorectal cancer (CRC) is among the most prevalent types of cancers that can be both identified and effectively treated if it is found in its earliest stages by screening, screening methods, such as colonoscopies and stool-based tests, are only effective to the extent that they are actually performed by patients. As a result, too many cases of CRC are diagnosed much later than they could be. This paper’s goal is identifying patients who are at higher risk of developing CRC earlier than otherwise using data currently being collected by the healthcare system about each individual patient.

The Electronic Health Record (EHR) is a collection of all of the data being collected from routine healthcare services provided to each patient. That data consists of lab test results, documented symptoms, and documentation of every visit made to a healthcare provider. Past research has investigated whether or not lab test abnormality findings such as anemia and gastrointestinal symptoms can be used to predict early risk of CRC [2]. Sadly, those patterns are usually non-specific and may not appear until it’s too late. Because of the potential for false positives, these patterns are limited for early risk detection when used alone. More recent research has used longitudinal EHR data to determine how a person’s data changes over time instead of just looking at one point in time [1]. Although using longitudinal data improves some predictions, lab and symptom data alone may not provide enough support for the early detection of CRC under tight constraints.
In addition to giving information on a patient's risk of developing CRC, EHRs also contain information about a patient's family history, their adherence to recommended screenings, and their patterns of diagnostics [4]. These factors might show an increased risk of disease before changes from the disease actually occur. The focus of this paper will be on those factors as they are documented in the EHR before a diagnosis and may show an increased risk of disease earlier than disease-related changes.
For the purpose of creating synthetic electronic health records (EHRs), Synthea was used for this paper. This allowed conducting large-scale experimentation to test hypotheses without having to use actual patient medical records. Using Synthea also has additional flexibility for testing and doesn’t present any ethical or privacy concerns compared to using real patient records.
This study hypothesizes that laboratory results and gastrointestinal symptoms alone are not enough for early colorectal cancer risk detection [1] when false-positives must be kept low. Incorporating longitudinal data, including family history and screening, will show an improvement in early risk detection.

PROJECT

Data
This paper uses electronic health records (EHRs) data generated with Synthea. Seven CSV files were given to the machine learning models, these include: patients.csv, conditions.csv, encounters.csv, observations.csv, procedures.csv, medications.csv, imaging_studies.csv. The data was kept in separate directories but was used simultaneously during the training of the model to make it scalable for future models which may contain over 40,000 patients.
Both datasets were kept in separate folders, but they were used simultaneously in order to make the project scalable for future models. This also ensures that future reiterations can have more than 40,000 patients.

Cohort and anchoring
Each patient with CRC was assigned an anchor date which is defined as the earliest CRC condition date. Control patients were sampled from non-CRC patients and were also given an anchor date. The final cohort contained 1,484 patients total (742 cases and 742 controls). 

Feature engineering
For each patient, there were monthly features or snapshots for up to 24 months before the anchor date. The monthly features included gastrointestinal symptoms signals, anemia-related signals from hemoglobin measurements, frequency signals, and simple counts (e.g., the number of GI conditions, encounters, and hemoglobin observations). One of the models had longitudinal risk context such as family history, screening indications (such as colonoscopies or stool screening), and diagnostic testing intensity as imaging-related signals. The output of this step is saved in crc_ml_table.csv.


Model
To train the models, the features were made before the anchor date. Summary statistics, such as max, min, mean, sum, for each feature were made from 1 to 12 months before the anchor date, or diagnosis. This created a single row for each patient that displayed their CRC status
In order to measure how well the models were doing, the study used two metrics: AUROC and AUPRC.  Since screening needs false-positives at a minimum, the model used a threshold to limit false-positives.
Since the hypothesis claims context will improve early detection, the study will compare two models:
    •    Lab values and symptom encounter features only
    •    Lab results and added risk context
The lab-only data models had low recall when the model used low false-positive constraints. On the other hand, adding risk context heavily improved early risk detection.

Results
AUROC measures how well the model ranks cancer patients above non-cancer patients. A value of 1.0 means perfect separation, while 0.5 represents random guessing. AUPRC shows how well the model balances detecting cancer while avoiding false alarms. A higher number shows the model has higher precision as recall increases. The difference between validation (VAL) and test (TEST) results is that validation results are numbers from data used to tune the model, while test results are from new data the model did not see before. Threshold (thr) is the probability cutoff used to see whether or not a patient is high-risk or labeled positive.  If a patient’s predicted probability is greater than or equal to the threshold, the model predicts cancer risk. Precision (prec) measures how many people who were flagged as high-risk actually had cancer. Recall (rec) is the fraction of actual cancer cases that were correctly identified. It answers, “Of all the patients who have cancer, how many did the model catch?” 
True positives (tp) shows the number of cancer patients that were correctly predicted as high-risk by the model. On the other hand, false-positives (fp) is the number of non-cancer patients that were incorrectly predicted as high-risk. A higher recall with less precision will lead to higher fp numbers. True negatives (tn) is the number of non-cancer patients that were correctly predicted as low-risk.  False-positive rate (fpr) is the fraction of control (non-cancer) patients that were incorrectly flagged as high-risk. It is calculated as FP / (FP + TN).
TEST thr=... shows different thresholds in precision, recall, and FPR. Having higher thresholds in machine learning models reduces false-positives but lower recall. By comparison, having lower thresholds should increase recall but it also increases false-positives since higher recall flags more people that might not have cancer.




![Figure 1](https://hc-cdn.hel1.your-objectstorage.com/s/v3/82604646772ab423_image.png)
Figure 1. Recall versus false-positive rate (FPR) for the laboratory-only colorectal cancer risk model evaluated on the test set.

![Figure 2](https://hc-cdn.hel1.your-objectstorage.com/s/v3/c3efa49f85819d05_image.png)
Figure 2. Recall versus false-positive rate (FPR) for the model using longitudinal risk context on the test set.


Figure 1 shows the relationship between recall and false-positive rates for the lab-only model across different thresholds. A lower threshold leads to an increased recall and a higher false-positive rate. This shows the complex relationship between finding more cases of cancer and mistakenly flagging patients as having cancer.
Figure 2 shows the results of the model with context and lab results.  Compared to the lab-only model, the context model reaches the same or even lower false positive rates with a bigger recall, especially within the 10% range where the number of false positives is limited. The two figures above both show how model performance changes as a threshold increases or decreases.

[Figure 3
](https://hc-cdn.hel1.your-objectstorage.com/s/v3/41dd8a483d47b99a_image.png)
Figure 3. This graph compares FPR and recall for lab-only (blue) and context  (orange) models on the test set. The blue dashed line shows FPR at 10%.

Figure 3 shows recall and false-positive rate for the lab-only and context models on the test set with multiple decision thresholds. Each point is a different threshold. This shows how recall changes as the false-positive rate increases. The graph also compares the two feature settings under the same controls. At an FPR of 10%, the context model has the higher recall at lower FPR than the laboratory-only model, but the lab-only model has a higher FPR to reach the same recall levels. This graph shows the difference in how the models balance both recall and false-positive rate with different thresholds.

CONCLUSION
This study was done to determine whether laboratory test results and gastrointestinal symptoms could reasonably identify people at early risk for colorectal cancer (CRC) and whether using longitudinal patient information would make identifying early CRC risk easier. The research supports this hypothesis. Models that were dependent solely on laboratory values and symptoms failed to meet the false positive thresholds during testing of new data. In contrast, models that utilized longitudinal risk information (i.e., screening history, patterns of diagnostic testing, etc.) were able to identify CRC cases while maintaining a lower level of false positives. These findings indicate that early CRC risk identification cannot be based solely on laboratory data and that a broader perspective is needed for improved early identification of CRC risk.

REFERENCES

Kennion, Oliver, et al. “Machine Learning as a New Horizon for Colorectal Cancer Risk Prediction? A Systematic Review.” Health Sciences Review, vol. 4, Sept. 2022, p. 100041, https://doi.org/10.1016/j.hsr.2022.100041. Accessed 26 Sept. 2022.
Făgărășan, Vlad, et al. “Absolute and Functional Iron Deficiency in Colon Cancer: A Cohort Study.” Medicina, vol. 58, no. 9, 1 Sept. 2022, p. 1202, https://doi.org/10.3390/medicina58091202.
Hussan, Hisham, et al. “Utility of Machine Learning in Developing a Predictive Model for Early-Age-Onset Colorectal Neoplasia Using Electronic Health Records.” PLOS ONE, vol. 17, no. 3, 10 Mar. 2022, p. e0265209, https://doi.org/10.1371/journal.pone.0265209. Accessed 5 Sept. 2022.
Sun, Chengkun, et al. “Predicting Early-Onset Colorectal Cancer in Individuals below Screening Age Using Machine Learning and Real-World Data: Case Control Study.” JMIR Cancer, vol. 11, 19 June 2025, pp. e64506–e64506, www.proquest.com/docview/3222950740?accountid=3783&sourcetype=Scholarly%20Journals, https://doi.org/10.2196/64506.
National Cancer Institute. “Colorectal Cancer Screening (PDQ®)–Health Professional Version - National Cancer Institute.” Www.cancer.gov, 4 June 2021, www.cancer.gov/types/colorectal/hp/colorectal-screening-pdq.
