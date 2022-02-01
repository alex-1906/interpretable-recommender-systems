# interpretable-recommender-systems

Please take the following steps to run the dashboards:

1. Clone the repository via SSH key or HTTP clone
2. Install the required dependencies running the following command: ```pip install -f requirements.txt```
3. Open terminal, navigate to the root folder of this repository and switch to the desired dashboard package e.g., ```cd content-based genres```, run the application with the following command:
```streamlit run main.py```       


The dashboard "association rules" needs to initially mine the global association rules. For this purpose please run the file association-rule-miner.py in the corresponding package.

For test purpose of different limitations and effects, the users 19 and 20 turned out to be useful.
