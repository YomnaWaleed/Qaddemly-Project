from data_preprocessing import *
from recommendation_to_user import *

user_df, jobs_df = preprocess('inputJson (1).json')
user_df = pd.read_pickle('user_df.pkl')
jobs_df = pd.read_pickle('jobs_df.pkl')

recommendations = recommend_for_user(user_df, jobs_df)
col_to_print = ['title', 'description', 'parsed_skills', 'location_type', 'employee_type', 'similarity_score']
for job in recommendations:
    print(user_df['skills'])
    for col in col_to_print:
        print(col.upper() + ": ", job[col])
    print("-------------------")