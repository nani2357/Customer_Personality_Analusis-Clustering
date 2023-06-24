import streamlit as st
import pandas as pd
import numpy as np
import pickle
from nbconvert import HTMLExporter
import nbformat
import streamlit.components.v1 as components
from PIL import Image

st.set_page_config(layout="wide")
# Load the model
def load_model():
    return pickle.load(open(r'D:\Customer_Personality_Analysis\final_model.sav', 'rb'))

# Make a prediction
def make_prediction(model, new_data):
    return model.predict(new_data)

def main():
    # Load the model
    model = load_model()
   
    # Sidebar
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ['Home','Insights & Cluster Analysis', 'Predict','Power BI Dashboard', 'Model Development','Model Flow Chart'])

    # Home page
    if page == 'Home':
        st.title('Customer Personality Analysis & Predictive Segmentation: A Real-World Application')


    
        st.markdown("""
        This Customer Personality Analysis (CPA) project is an illustration of my real-time work experience. It encapsulates the process and approach I have utilized in professional settings, making it an important asset in my portfolio. The primary objective is to leverage data analysis and machine learning techniques to understand and segment a company's diverse customer base. The ultimate goal is to support the development of targeted marketing strategies and enhance customer engagement.
    
        In the professional project, the workflow is organized into four stages:
    
        1. Customer Sentiment Analysis
        2. Customer Segmentation
        3. Predictive Model Development for Future Segmentation
        4. Product Recommendations
    
        This portfolio project will focus on the second(Customer Segmentation) and third stages(Predictive Model Development for Future Segmentation), demonstrating my proficiency in customer segmentation and developing predictive models for future data segmentation.
        """)
        
        st.header("Real-World Application and Audience")
        st.markdown("""
        The techniques and methodologies used in this Customer Personality Analysis project are not limited to a theoretical context but find numerous applications in the real world across various industries. The predictive models and segmentation approaches demonstrated here are particularly useful for several stakeholders:
        
        **1. Marketing and Strategy Teams**: For marketing professionals, understanding customer behavior is paramount. For example, an e-commerce business might want to target customers who are more likely to make repeat purchases. By segmenting customers based on their purchasing behaviors and other characteristics, tailored marketing strategies can be designed. This would result in more effective marketing campaigns, potentially leading to increased customer conversion rates and business growth.
        
        **2. Product Development Teams**: In the realm of product development, customer insights drive innovation. Consider a tech company that develops mobile applications. By understanding the distinct needs and preferences of different customer segments, the product team can focus on features that resonate with specific user groups. This can expedite product development cycles, align the product better with market demands, and consequently, boost customer satisfaction and brand reputation.
        
        **3. Customer Engagement Teams**: Enhancing customer experience is at the heart of any customer-centric business. For instance, a streaming service like Netflix could use the insights derived from this project to deliver personalized experiences. By understanding customer segments, Netflix can recommend content based on viewers' individual preferences, thereby increasing user engagement and enhancing customer loyalty.
        
        In the professional world, businesses across industries, be it retail, tech, or entertainment, employ data science to understand their customers and improve their product or service offerings. They use similar techniques to segment their customers, predict future behaviors, and develop personalized marketing strategies. 
        
        This project serves as a practical example of my real-world work in professional settings. It encapsulates the process and approach I have utilized in actual projects, making it an important asset in my portfolio. It showcases my ability to work with data, derive actionable insights, and build predictive models that can drive strategic business decisions.
        """)

        st.header("Introduction of the Project")
        st.markdown("""
        My Customer Personality Analysis (CPA) project represents a crucial aspect of modern data-driven decision-making within the business context. Drawing from real-life experiences, this project serves to address the complexities and diversity within a company's customer base. Through machine learning and data analysis techniques, we aim to categorize customers into distinct segments based on their unique behaviors and characteristics. The goal of this project is to facilitate targeted marketing strategies, enhancing customer engagement, and promoting effective product recommendations.
        """)
    
        st.subheader("Project Stages")
        st.markdown("""
        The project will proceed in two major stages, mirroring real-life project execution:
    
        **Customer Segmentation**: This stage involves the use of clustering algorithms to classify customers into different groups based on their distinct attributes and purchasing behaviors. The derived segments will serve as a foundation for tailored marketing strategies and decision-making processes.
    
        **Model Development for Future Data**: Building on the customer segmentation, this stage focuses on developing a predictive model that can handle future data. This model will enable the company to anticipate changes in customer behavior and adapt their strategies accordingly.
        """)
    
        st.subheader("Business Requirements")
        st.markdown("""
        1. The core business requirement is to provide an analytical foundation that enables marketing and strategy teams to identify and target specific customer segments. Key business needs include:
    
        2. Enabling targeted marketing campaigns by identifying customer segments more likely to respond positively.
        3. Streamlining product development by understanding the unique needs and preferences of different customer segments.
        3. Enhancing customer engagement by delivering personalized experiences based on individual customer segments.
        """)
    
        st.subheader("Proposed Solution")
        st.markdown("""
        My solution focuses on leveraging machine learning techniques to meet the defined business requirements:
    
        **Data Analysis and Clustering**: Conduct extensive data preprocessing and exploratory analysis to identify key features. Use clustering algorithms to create distinct customer segments.
    
        **Predictive Model Development**: Post-segmentation, develop a robust predictive model using machine learning algorithms to anticipate future customer behaviors and preferences.
    
        This solution aims to equip marketing teams with actionable insights that enhance customer satisfaction and drive business performance. By identifying customer segments and predicting future behaviors, companies can create more targeted and effective marketing strategies.
        """)
    
        st.subheader("Approach")
        st.markdown("""
        Our approach to this project follows a structured, step-by-step methodology grounded in data science best practices. Each stage is thoughtfully designed to build upon the previous, ensuring a cohesive and comprehensive solution.
    
        **Understanding the Data**: The first step involves a thorough understanding of the dataset, its variables, and its structure. This step is crucial for shaping the subsequent stages of the project.
    
        **Data Preprocessing**: After understanding the dataset, we clean and preprocess the data. This involves handling missing values, potential outliers, and categorical variables, ensuring the data is ready for analysis.
    
        **Exploratory Data Analysis (EDA)**: This stage involves unearthing patterns, spotting anomalies, testing hypotheses, and checking assumptions through visual and quantitative methods. It provides an in-depth understanding of the variables and their interrelationships, which aids in feature selection.
    
        **Feature Selection**: Based on the insights from EDA, relevant features are selected for building the machine learning model. Feature selection is critical to improve the model's performance by eliminating irrelevant or redundant information.
    
        **Customer Segmentation**: The preprocessed data is then fed into a clustering algorithm to group customers into distinct segments based on their attributes and behavior. This segmentation enables targeted marketing and personalized customer engagement.
    
        **Model Development**: Once we have our customer segments, we develop a predictive model using a suitable machine learning algorithm. This model is trained on the current data and then validated using a separate test set.
    
        **Model Evaluation and Optimization**: The model's performance is evaluated using appropriate metrics. If necessary, the model is fine-tuned and optimized to ensure the best possible performance.
    
        **Prediction on Future Data**: The final step involves utilizing the trained model to make predictions on future data. This will allow the business to anticipate changes in customer behavior and adapt their strategies accordingly.
    
        This approach ensures a systematic and thorough analysis of the customer data, leading to robust and reliable customer segments and predictions. It aims to provide a foundation upon which strategic business decisions can be made and future customer trends can be anticipated.
        """)
    
        st.subheader("Conclusion")
        st.markdown("""
        The purpose of the presented analysis was to develop an effective predictive model that can identify the target customers. Three separate experiments were conducted with varying features and data manipulation techniques, which included SMOTE for handling class imbalance.
    
        The experiments compared a set of classification models including logistic regression, Naive Bayes, decision tree, random forest, SVM, K-Nearest Neighbors (KNN), gradient boosting (XGB Classifier), neural networks (MLP), and AdaBoost. Among these, the gradient boosting model, specifically XGBoost, consistently performed the best across all experiments.
    
        Experiment 3, which incorporated nearly all features and applied SMOTE for oversampling, was the most successful in terms of performance metrics, implying that both a more comprehensive feature set and balanced data contribute positively to the model's performance.
    
        Moreover, XGBoost's performance was verified by splitting the data into training, validation, and test sets. Hyperparameter optimization was also performed to ensure that the best parameters were selected for the final model. This was initially done with GridSearchCV, but due to computational constraints, RandomizedSearchCV was utilized as a more efficient alternative.
    
        The accuracy scores from both validation and test data demonstrate that the XGBoost model generalized well, reducing the likelihood of overfitting. Confusion matrix, learning curve, and class prediction error plots further confirmed the model's good performance.
    
        Based on the model development experiments 1, 2, and 3, the following conclusions can be made:
    
        **Feature Selection**: The chosen experiment was Experiment 3, which involved the use of almost all features and the application of the Synthetic Minority Over-sampling Technique (SMOTE) to handle class imbalance. Despite differences in feature selection and handling of class imbalance among the experiments, all models performed similarly. This suggests that using a more comprehensive set of features, as in Experiment 3, provided a well-fitted input for the classification models without negatively impacting performance.
    
        **SMOTE's Impact**: Experiment 3, which made use of SMOTE to balance the classes, was chosen over Experiment 2 which did not use SMOTE. This decision underscores the importance of handling class imbalance in the dataset. While the impact of SMOTE was not drastically apparent in the difference between the performances of the models, its utilization is critical in datasets with imbalanced classes to ensure that the minority class is not ignored.
    
        **Model Performance**: Among all the classification algorithms tested, the Gradient Boosting method, specifically the XGBoost Classifier, showed the best performance with fewer errors in the Class Prediction Error plot. It accurately identified the main target customers (class 1), which was the objective of this exercise.
    
        Given these findings, the XGBoost Classifier was chosen for further hyperparameter tuning. This process of optimization helps enhance the performance of the model by adjusting the model parameters to their ideal values. Once this is achieved, the model will be saved and deployed. The deployment of the model allows it to be used in practical applications, providing predictions on new, unseen data.
        In conclusion, the analysis underscores the importance of careful feature selection, handling of class imbalance, and selection of the right model in creating robust and accurate predictive models. The chosen experiment and model, Experiment 3 and the XGBoost Classifier respectively, were the best fit for this dataset and problem, providing reliable predictions and highlighting the main target customer group (class 1) accurately. It also indicates the importance of model optimization via hyperparameter tuning to maximize the performance of the chosen model.
    
        To conclude, the XGBoost model developed in this analysis has shown to be a strong predictor for the target customers. It has a good balance between bias and variance, which makes it a reliable tool for new, unseen data. Therefore, this model was chosen as the final model, and will be saved and deployed for further use.
        """)
        
        
    if page == 'Insights & Cluster Analysis':
        
        st.title('Harnessing Customer Segmentation for Strategic Marketing')

        # Subheader
        st.subheader('Insights and Recommendations based on Cluster Analysis')
        
        # Detailed content in Markdown
        st.markdown("""
        ## Overview
        
        Post an extensive exploratory data analysis and insightful visualizations via Power BI, the stakeholders and marketing strategy teams have chosen to progress with the PCA1 cluster group. The choice was influenced by the group's superior capability to underline underlying patterns and segregate the data into meaningful segments.
        
        This process revealed several interconnected relationships between features that offer valuable insights for targeted marketing and customer engagement strategies. These insights, originating from the PCA1 cluster group analysis, will substantially shape the teams' future course of action and assist in designing personalized, data-driven marketing campaigns.
        
        ## Customer Clusters and Strategic Recommendations
        
        ### Cluster 0: High Spending Singles
        
        This group typically has higher income, usually exceeding 80K. Their high purchasing power is evident in the spending range of 1300 to 2000. They primarily consist of single individuals, most of whom do not have children or only one child. This segment shows high engagement with campaigns, a majority having accepted campaign offerings in the past.
        
        #### Business Recommendations
        
        Given their high income and spending, marketing premium and exclusive products/services could resonate well with this group. Personalized and time-limited offers could be particularly effective considering their high engagement with past campaigns.
        
        ### Cluster 1: Low-Income Families
        
        This cluster comprises lower-income individuals, typically earning between 0 to 50K. Their spending capacity is comparably lower, falling in the range of 10 to 250. Predominantly, this group consists of couples, most of whom have at least 1 or 2 children. Many individuals in this group are educated, and the family size is typically larger, ranging from 3 to 5 with no singles. The age range is quite mature, from 40 to 60 years old.
        
        #### Business Recommendations
        
        Offering discounts, family-centric packages, and educational resources might be appealing to this group. Considering their lower spending capacity and larger family size, cost-effective products or services and family deals may also find favor.
        
        ### Cluster 2: Middle-Income Single Parents
        
        This cluster comprises a middle-income group, with earnings falling between 50K and 80K. Their spending is average, with a range of 1000 to 1600. They are predominantly single parents with only one child, indicating a smaller family size not exceeding 3 members.
        
        #### Business Recommendations
        
        Given their balanced income and spending, this group may respond well to value-for-money offerings. Marketing strategies could include single parent-focused campaigns or promotions for moderately priced items, perhaps with an emphasis on quality and longevity.
        
        **Note:** The cluster group numbers may change with different runs of the clustering algorithm, but the core characteristics of the clusters remain consistent.
        """)





    # Prediction page
    elif page == 'Predict':
        st.title('Predict the Customer Class')


                # Inputs
        Income = st.number_input('Income')
        Kidhome = st.number_input('Kidhome')
        Teenhome = st.number_input('Teenhome')
        Recency = st.number_input('Recency')
        MntWines = st.number_input('MntWines')
        MntFruits = st.number_input('MntFruits')
        MntMeatProducts = st.number_input('MntMeatProducts')
        MntFishProducts = st.number_input('MntFishProducts')
        MntSweetProducts = st.number_input('MntSweetProducts')
        MntGoldProds = st.number_input('MntGoldProds')
        NumDealsPurchases = st.number_input('NumDealsPurchases')
        NumWebPurchases = st.number_input('NumWebPurchases')
        NumCatalogPurchases = st.number_input('NumCatalogPurchases')
        NumStorePurchases = st.number_input('NumStorePurchases')
        NumWebVisitsMonth = st.number_input('NumWebVisitsMonth')
        
        AcceptedCmp1 = st.selectbox('AcceptedCmp1', ['Yes', 'No'])
        AcceptedCmp1 = 1 if AcceptedCmp1 == 'Yes' else 0
        
        AcceptedCmp2 = st.selectbox('AcceptedCmp2', ['Yes', 'No'])
        AcceptedCmp2 = 1 if AcceptedCmp2 == 'Yes' else 0
        
        AcceptedCmp3 = st.selectbox('AcceptedCmp3', ['Yes', 'No'])
        AcceptedCmp3 = 1 if AcceptedCmp3 == 'Yes' else 0
        
        AcceptedCmp4 = st.selectbox('AcceptedCmp4', ['Yes', 'No'])
        AcceptedCmp4 = 1 if AcceptedCmp4 == 'Yes' else 0
        
        AcceptedCmp5 = st.selectbox('AcceptedCmp5', ['Yes', 'No'])
        AcceptedCmp5 = 1 if AcceptedCmp5 == 'Yes' else 0
        
        label_map = {'Basic': 1, 'Graduation': 2, 'Master': 3, 'PhD': 4}
        Education_encode = st.selectbox('Education', list(label_map.keys()))
        Education_encode = label_map[Education_encode]
        
        Complain = st.selectbox('Complain', ['Yes', 'No'])
        Complain = 1 if Complain == 'Yes' else 0
        
        Response = st.selectbox('Response', ['Yes', 'No'])
        Response = 1 if Response == 'Yes' else 0
        
        Age = st.number_input('Age')
        Customer_Since_Years = st.slider('Customer_Since_Years', 0, 100)
        
        # Calculate computed features
        Total_spend = MntWines + MntFruits + MntMeatProducts + MntFishProducts + MntSweetProducts + MntGoldProds
        Total_purchase = NumDealsPurchases + NumWebPurchases + NumCatalogPurchases + NumStorePurchases
        Total_children = Kidhome + Teenhome
        Total_adults = Total_children # Assuming Total_adults = Total_children (Adjust as needed)
        Accepted_camp = AcceptedCmp1 + AcceptedCmp2 + AcceptedCmp3 + AcceptedCmp4 + AcceptedCmp5
        Family_size = Total_adults + Total_children
        
        # Collect data into a DataFrame
        data = {
            'Income': [Income],
            'Kidhome': [Kidhome],
            'Teenhome': [Teenhome],
            'Recency': [Recency],
            'MntWines': [MntWines],
            'MntFruits': [MntFruits],
            'MntMeatProducts': [MntMeatProducts],
            'MntFishProducts': [MntFishProducts],
            'MntSweetProducts': [MntSweetProducts],
            'MntGoldProds': [MntGoldProds],
            'NumDealsPurchases': [NumDealsPurchases],
            'NumWebPurchases': [NumWebPurchases],
            'NumCatalogPurchases': [NumCatalogPurchases],
            'NumStorePurchases': [NumStorePurchases],
            'NumWebVisitsMonth': [NumWebVisitsMonth],
            'AcceptedCmp3': [AcceptedCmp3],
            'AcceptedCmp4': [AcceptedCmp4],
            'AcceptedCmp5': [AcceptedCmp5],
            'AcceptedCmp1': [AcceptedCmp1],
            'AcceptedCmp2': [AcceptedCmp2],
            'Complain': [Complain],
            'Response': [Response],
            'Age': [Age],
            'Total_spend': [Total_spend],
            'Total_purchase': [Total_purchase],
            'Total_children': [Total_children],
            'accepted_camp': [Accepted_camp],
            'Total_adults': [Total_adults],
            'Family_size': [Family_size],
            'Customer_Since_Years': [Customer_Since_Years],
            'Education_encode': [Education_encode]
        }
        new_data = pd.DataFrame(data)
        if st.button('Predict'):
            prediction = make_prediction(model, new_data)
            st.write('The predicted class is ', prediction[0])
            
    elif page == "Power BI Dashboard":
        
        st.title('Power BI Dashboard')
        report_url = "https://app.powerbi.com/view?r=eyJrIjoiNGFhN2Y0ODktZTVlYi00NWI4LWIwZWUtZGJjNDM0Nzk1N2JjIiwidCI6ImRmODY3OWNkLWE4MGUtNDVkOC05OWFjLWM4M2VkN2ZmOTVhMCJ9"  
        components.html(f'<iframe width="1000" height="600" src="{report_url}"></iframe>', width=1100, height=1000)
            
    elif page == "Model Development":
        st.markdown("<h2 style='text-align: left; color: black;'> End-to-End Model Development: Harnessing Clustering and Classification Techniques</h4>", unsafe_allow_html=True)
        
        # Read the Jupyter notebook file
        with open(r'D:\Customer_Personality_Analysis\final_model.ipynb', 'r') as f:
            notebook = nbformat.read(f, as_version=4)

    # Convert the Jupyter notebook to HTML
        html_exporter = HTMLExporter()
        html_exporter.template_name = 'classic'
        (body, _) = html_exporter.from_notebook_node(notebook)
        components.html(body,width=1000, height=1200, scrolling=True)
        
    elif page == "Model Flow Chart":
        st.title('Model Flow Chart')
        image = Image.open("D:/Customer_Personality_Analysis/model_folowchart.png")
        st.image(image, use_column_width=True)



            
if __name__ == "__main__":
    main()