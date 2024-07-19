# Credit-Risk-Assessment
This project make my team a champion of The Data Analytics Competition 2024, organized by Ho Chi Minh university of Banking.
## Topic

This project builds a machine learning model to assess user credit risk.

## Data Information

This project, we use internal bank data and data from the Credit Information Center (CIC). Information about the two datasets is described as follows:

**train_1:** Internal Bank Data

| Variable             | Description                                         |
| -------------------- | --------------------------------------------------- |
| Case_ID              | Customer ID                                         |
| Total_TL             | Total accounts in the Credit Information Center     |
| Tot_Closed_TL        | Total closed accounts                               |
| Tot_Active_TL        | Total active accounts                               |
| Total_TL_opened_L6M  | Total accounts opened in the last 6 months          |
| Tot_TL_closed_L6M    | Total accounts closed in the last 6 months          |
| pct_tl_open_L6M      | Percentage of accounts opened in the last 6 months  |
| pct_tl_closed_L6M    | Percentage of accounts closed in the last 6 months  |
| pct_active_tl        | Percentage of active accounts                       |
| pct_closed_tl        | Percentage of closed accounts                       |
| Total_TL_opened_L12M | Total accounts opened in the last 12 months         |
| Tot_TL_closed_L12M   | Total accounts closed in the last 12 months         |
| pct_tl_open_L12M     | Percentage of accounts opened in the last 12 months |
| pct_tl_closed_L12M   | Percentage of accounts closed in the last 12 months |
| Tot_Missed_Pmnt      | Total missed payments                               |
| Auto_TL              | Number of auto loans                                |
| CC_TL                | Number of credit card loans                         |
| Consumer_TL          | Number of consumer goods loans                      |
| Gold_TL              | Number of gold loans                                |
| Home_TL              | Number of home loans                                |
| PL_TL                | Number of personal loans                            |
| Secured_TL           | Number of secured accounts                          |
| Unsecured_TL         | Number of unsecured accounts                        |
| Other_TL             | Number of other accounts                            |
| Age_Oldest_TL        | Age of the oldest account (years)                   |
| Age_Newest_TL        | Age of the newest account (years)                   |

**train_2.csv:** Credit Information Center (CIC) Data

| Variable                     | Description                                                                  |
| ---------------------------- | ---------------------------------------------------------------------------- |
| Case_id                      | Customer ID                                                                  |
| time_since_recent_payment    | Time since the most recent payment                                           |
| time_since_first_deliquency  | Time since the first delinquency                                             |
| time_since_recent_deliquency | Time since the most recent delinquency                                       |
| num_times_delinquent         | Number of delinquencies                                                      |
| max_delinquency_level        | Highest delinquency level                                                    |
| max_recent_level_of_deliq    | Highest recent delinquency level                                             |
| num_deliq_6mts               | Number of delinquencies in the last 6 months                                 |
| num_deliq_12mts              | Number of delinquencies in the last 12 months                                |
| num_deliq_6_12mts            | Number of delinquencies between 6 and 12 months                              |
| max_deliq_6mts               | Highest delinquency level in the last 6 months                               |
| max_deliq_12mts              | Highest delinquency level in the last 12 months                              |
| num_times_30p_dpd            | Number of times 30 days past due                                             |
| num_times_60p_dpd            | Number of times 60 days past due                                             |
| num_std                      | Total number of standard on-time payments                                    |
| num_std_6mts                 | Number of on-time payments in the last 6 months                              |
| num_std_12mts                | Number of on-time payments in the last 12 months                             |
| num_sub                      | Number of substandard payments                                               |
| num_sub_6mts                 | Number of substandard payments in the last 6 months                          |
| num_sub_12mts                | Number of substandard payments in the last 12 months                         |
| num_dbt                      | Number of doubtful payments                                                  |
| num_dbt_6mts                 | Number of doubtful payments in the last 6 months                             |
| num_dbt_12mts                | Number of doubtful payments in the last 12 months                            |
| num_lss                      | Number of loss accounts                                                      |
| num_lss_6mts                 | Number of loss accounts in the last 6 months                                 |
| num_lss_12mts                | Number of loss accounts in the last 12 months                                |
| recent_level_of_deliq        | Most severe recent delinquency level                                         |
| tot_enq                      | Total number of credit inquiries                                             |
| CC_enq                       | Total number of credit card inquiries                                        |
| CC_enq_L6m                   | Number of credit card inquiries in the last 6 months                         |
| CC_enq_L12m                  | Number of credit card inquiries in the last 12 months                        |
| PL_enq                       | Total number of personal loan inquiries                                      |
| PL_enq_L6m                   | Number of personal loan inquiries in the last 6 months                       |
| PL_enq_L12m                  | Number of personal loan inquiries in the last 12 months                      |
| time_since_recent_enq        | Time since the most recent inquiry                                           |
| enq_L12m                     | Number of inquiries in the last 12 months                                    |
| enq_L6m                      | Number of inquiries in the last 6 months                                     |
| enq_L3m                      | Number of inquiries in the last 3 months                                     |
| MARITALSTATUS                | Marital status                                                               |
| EDUCATION                    | Education level                                                              |
| AGE                          | Age                                                                          |
| GENDER                       | Gender                                                                       |
| NETMONTHLYINCOME             | Net monthly income                                                           |
| Time_With_Curr_Empr          | Time with current employer                                                   |
| pct_of_active_TLs_ever       | Percentage of ever-active accounts to total opened                           |
| pct_opened_TLs_L6m_of_L12m   | Percentage of accounts opened between 6 and 12 months                        |
| pct_currentBal_all_TL        | Percentage of current balance to total credit limit                          |
| CC_utilization               | Credit card utilization ratio                                                |
| CC_Flag                      | Credit card flag (0: Normal, 1: Issue)                                       |
| PL_utilization               | Personal loan utilization ratio                                              |
| PL_Flag                      | Personal loan flag (0: Normal, 1: Issue)                                     |
| pct_PL_enq_L6m_of_L12m       | Percentage of personal loan inquiries from 6 to 12 months                    |
| pct_CC_enq_L6m_of_L12m       | Percentage of credit card inquiries from 6 to 12 months                      |
| pct_PL_enq_L6m_of_ever       | Percentage of personal loan inquiries in the last 6 months to ever inquiries |
| pct_CC_enq_L6m_of_ever       | Percentage of credit card inquiries in the last 6 months to ever inquiries   |
| max_unsec_exposure_inPct     | Maximum percentage of total unsecured credit exposure                        |
| HL_Flag                      | Home loan flag                                                               |
| GL_Flag                      | Gold loan flag                                                               |
| last_prod_enq2               | Most recent product inquiry                                                  |
| first_prod_enq2              | First product inquiry                                                        |
| Approved_Flag                | Customer classification                                                      |

## Table of contents

* Data Loading
* EDA (Exploratory Data Analysis)
* Feature Engineering
* Model Building
* Evaluating

## Consulutionsss

1. In this project, we use XGBoost to perform customer risk classification.The results are as follows: Accuracy: 0.954, F1-Score: 0.954, AUC: 0.969
2. Additionally, we introduce an approach to effectively build and tackle a classification problem.
3. Future work includes applying other algorithms and tuning parameters to achieve the highest possible efficiency.
