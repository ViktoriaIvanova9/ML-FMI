import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Lasso, Ridge, LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

def analyse_types(fitgpt_df):
    print()
    print(fitgpt_df.info)
    print(fitgpt_df.dtypes)

    print(f'Gender unique values: {np.unique(fitgpt_df['Gender'].values)}')
    print(f'Workout type unique values: {np.unique(fitgpt_df['Workout_Type'].values)}')
    print()

def statistical_data(fitgpt_df):
    # characteristics - mean, median, mode, standard deviation, histogram

    for label, data in fitgpt_df.items():
        if data.dtypes != 'object':
            print(f'Mean value for {label} : {np.round(np.mean(data), 3)}')
            print(f'Median value for {label} : {np.round(np.median(data), 3)}')
            print(f'Standard deviation for {label} : {np.round(np.std(data), 3)}')

            plt.title(f'{label}')
            plt.hist(data, bins=10)
            plt.show()
        else:
            _, counts = np.unique(data, return_counts=True)
            index_of_most_freq = np.argmax(counts)
            print(f'Mode value for {label} : {data[index_of_most_freq]}')

        print()
    
    # Categorical features bar plot evaluation
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    f.suptitle("Histogram of non-numerical features in FitGPT DataFrame")
    ax1.set_title("Gender")
    ax1.set_ylabel("Number of people")
    gender_types = np.unique(fitgpt_df['Gender'].values)
    male_count = len(fitgpt_df.loc[fitgpt_df['Gender'] == 'Male'])
    female_count = len(fitgpt_df.loc[fitgpt_df['Gender'] == 'Female'])
    ax1.bar(gender_types, [male_count, female_count] , width=0.35)

    ax2.set_title("Workout_Type")
    workout_types = np.unique(fitgpt_df['Workout_Type'].values)
    number_trainees = []
    for workout in workout_types:
        number_trainees.append(len(fitgpt_df[fitgpt_df['Workout_Type'] == workout]))
    ax2.bar(workout_types, number_trainees , width=0.35)

    plt.tight_layout()
    plt.show()

def missing_values(fitgpt_df):
    # check the percent of missing values 
    missing_values = fitgpt_df.isna().mean().sort_values(ascending=False)
    print(missing_values)
    # There are no missing values in this data

    # removing rows from columns where the missing values are < 5%
    drop_from_columns = missing_values[missing_values < 0.05].index.to_list()
    fitgpt_df = fitgpt_df.dropna(subset=drop_from_columns)

    # removing columns where the missing data is > 65%
    drop_columns = missing_values[missing_values > 0.65]
    fitgpt_df = fitgpt_df.dropna(axis=1, subset=drop_columns)

    # imputing missing values with the best value
    X = fitgpt_df.drop('Calories_Burned', axis=1)
    y = fitgpt_df['Calories_Burned'].values

    imputer = SimpleImputer(strategy='most_frequent')
    imputer.fit(X)
    X_transformed = imputer.transform(X)
    print(f'Data after imputing the missing values:\n {X_transformed}')
    # default strategy - mean. Other strategies - median, most_frequent, constant

def extremal_values(fitgpt_df):
    # defining extremal values
    print(fitgpt_df.describe().T)

    # scaling the data
    X = fitgpt_df.drop(['Gender', 'Workout_Type'], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f'Standard deviation before scaling: {np.std(X['Calories_Burned'])}')
    print(f'Standard deviation after scaling: {np.std(X_scaled)}')

    # boxplots analysing the extremal data
    fitgpt_df = fitgpt_df.drop(['Gender', 'Workout_Type'], axis=1)

    for label, data in fitgpt_df.items():
        plt.title(f'Analysing extremal values for {label}')
        plt.boxplot(data)
        plt.tight_layout()
        plt.show()

    # the data can be normalized or standardized

# ====================================================
#                   Data exploration
# ====================================================

def correlation_heatmap_analysis(fitgpt_df):
    fitgpt_df = fitgpt_df.drop(['Gender', 'Workout_Type'], axis=1)

    fit_corr=fitgpt_df.corr(method='pearson')
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(fit_corr, interpolation='nearest')

    fig.colorbar(im, orientation='vertical', fraction = 0.05)
    ax.set_xticks(range(0,13))
    ax.set_xticklabels(fitgpt_df.columns, rotation='vertical')
    ax.set_yticks(range(0,13), fitgpt_df.columns)

    plt.tight_layout()
    plt.show()

def calories_per_workout(fitgpt_df):
    plt.title("Calories burned based on the workout type")
    workout_types = np.unique(fitgpt_df['Workout_Type'].values)
    mean_calories_per_type = {}
    for workout in workout_types:
        mean_calories_per_type[workout] = np.mean(fitgpt_df[fitgpt_df['Workout_Type'] == workout]['Calories_Burned'])

    plt.xlabel("Workout_Type")
    plt.ylabel("Calories_Burned")
    plt.plot(workout_types, mean_calories_per_type.values())
    plt.tight_layout()
    plt.show()

def fat_to_weight(fitgpt_df):
    plt.title("Fat Percentage of person, based on their weight")
    plt.scatter(fitgpt_df['Weight (kg)'], fitgpt_df['Fat_Percentage'])
    plt.xlabel('Weight (kg)')
    plt.ylabel('Fat_Percentage')
    plt.tight_layout()
    plt.show()

def frequency_to_experience(fitgpt_df):
    plt.title("Experience based on how many days a week they train")
    num_days = np.arange(1, 7)
    experience = {}
    for days in num_days:
        experience[days] = np.mean(fitgpt_df[fitgpt_df['Workout_Frequency (days/week)'] == days]['Experience_Level'])
    plt.bar(num_days, experience.values())
    plt.xlabel('Workout_Frequency')
    plt.ylabel('Experience_Level')
    plt.tight_layout()
    plt.show()

def weight_bmi(fitgpt_df):
    plt.title("Weight dependence on BMI")
    plt.scatter(fitgpt_df['Weight (kg)'], fitgpt_df['BMI'])
    plt.xlabel('Weight (kg)')
    plt.ylabel('BMI')
    plt.tight_layout()
    plt.show()

def duration_calories(fitgpt_df):
    plt.title("Calories burned, based on the duration of the session")
    plt.scatter(fitgpt_df['Session_Duration (hours)'], fitgpt_df['Calories_Burned'])
    plt.xlabel('Session_Duration')
    plt.ylabel('Calories_Burned')
    plt.tight_layout()
    plt.show()

def BPM_by_Age(fitgpt_df):
    plt.title("Age - BPM")
    min_age = np.min(fitgpt_df['Age'])
    max_age = np.max(fitgpt_df['Age'])
    resting_bpm = {}
    avg_bpm = {}
    max_bpm = {}

    for age in range(min_age, max_age-5, 5):
        resting_bpm[age] = np.mean(fitgpt_df.loc[fitgpt_df['Age'].between(age, age + 5)]['Resting_BPM'])
        avg_bpm[age] = np.mean(fitgpt_df.loc[fitgpt_df['Age'].between(age, age + 5)]['Avg_BPM'])
        max_bpm[age] = np.mean(fitgpt_df.loc[fitgpt_df['Age'].between(age, age + 5)]['Max_BPM'])
    
    plt.plot(resting_bpm.keys(), resting_bpm.values())
    plt.plot(avg_bpm.keys(), avg_bpm.values())
    plt.plot(max_bpm.keys(), max_bpm.values())
    plt.xlabel('Age slot')
    plt.ylabel('BPM')
    plt.legend(['Resting', 'Average', 'Max'])
    plt.tight_layout()
    plt.show()

def gender_workout_type(fitgpt_df):
    plt.title('Workout type based on gender')
    workout_types = np.unique(fitgpt_df['Workout_Type'].values)

    workout_by_gender = {}
    for workout in workout_types:
        male_choice_cnt = len(fitgpt_df.loc[(fitgpt_df['Workout_Type'] == workout) & (fitgpt_df['Gender'] == 'Male')])
        female_choice_cnt = len(fitgpt_df.loc[(fitgpt_df['Workout_Type'] == workout) & (fitgpt_df['Gender'] == 'Female')])
        workout_by_gender[workout] = [male_choice_cnt, female_choice_cnt]

    plt.plot(workout_types, workout_by_gender.values())
    plt.legend(['Male', 'Female'])
    plt.tight_layout()
    plt.show()

def fat_experience_level(fitgpt_df):
    plt.title("Fat percentage, based on the experience level")
    plt.bar(fitgpt_df['Experience_Level'], fitgpt_df['Fat_Percentage'])
    plt.xlabel('Experience_Level')
    plt.ylabel('Fat_Percentage')
    plt.tight_layout()
    plt.show()

def attributes_choose(fitgpt_df):
    pass
    # linearly independent attributes which will give the most information without "confusing" the model
    # Age                           
    # Gender                           
    # Weight (kg)                     
    # Height (m)                       
    # Max_BPM                          
    # Avg_BPM                          
    # Resting_BPM                      
    # Session_Duration (hours)         
    # Calories_Burned                  
    # Workout_Type                     
    # Fat_Percentage                   
    # Water_Intake (liters)            
    # Workout_Frequency (days/week)    
    # Experience_Level             
    # BMI                              

def train_and_evaluate_data(fitgpt_df):
    pass

def knn(fitgpt_df):
    fitgpt_dummies = pd.get_dummies(fitgpt_df['Gender'], drop_first=True, dtype=int)
    fitgpt_dummies = pd.concat([fitgpt_df, fitgpt_dummies], axis=1)
    fitgpt_dummies = fitgpt_dummies.drop(columns=['Gender'])


    X = fitgpt_dummies[['Age', 'BMI', 'Session_Duration (hours)', 'Water_Intake (liters)', 'Male']]
    y = fitgpt_dummies['Calories_Burned'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    test_accuracies = {}

    neighbours = np.arange(1, 13)
    for n_neighbours_cnt in neighbours:
        knn = KNeighborsClassifier(n_neighbors=n_neighbours_cnt)
        knn.fit(X_train, y_train)

        test_accuracies[n_neighbours_cnt] = round(knn.score(X_test, y_test), 4)

    print(f'Score based on the number of neighbours: {test_accuracies}')
    plt.plot(test_accuracies.keys(), test_accuracies.values())

    plt.title('KNN: Varying Number of Neighbors')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.show()

def log_reg(fitgpt_df):
    fitgpt_dummies = pd.get_dummies(fitgpt_df['Gender'], drop_first=True, dtype=int)
    fitgpt_dummies = pd.concat([fitgpt_df, fitgpt_dummies], axis=1)
    fitgpt_dummies = fitgpt_dummies.drop(columns=['Gender'])

    X = fitgpt_dummies[['Age', 'BMI', 'Session_Duration (hours)', 'Water_Intake (liters)', 'Male']]
    y = fitgpt_dummies['Calories_Burned'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    parameters = {'logreg__C': np.linspace(0.001, 1.0, 20)}

    steps = [('scaler', StandardScaler()),
            ('logreg', LogisticRegression(random_state=42))]
    pipeline = Pipeline(steps)
    logreg = GridSearchCV(pipeline, parameters).fit(X_train, y_train)
    logreg_score = logreg.score(X_test, y_test)

    print(f'With scaling: {logreg_score}')
    print(f'With scaling: {logreg.best_params_}')

def main():
    fitgpt_df = pd.read_csv("calories_burned_fitgpt_gym.csv")

    # analyse_types(fitgpt_df)
    # statistical_data(fitgpt_df)
    # missing_values(fitgpt_df)
    # extremal_values(fitgpt_df)

    # Analyzing the data and choosing the attributes

    # correlation_heatmap_analysis(fitgpt_df)
    # calories_per_workout(fitgpt_df) # HIIT - most calories burned, Cardio - least
    # fat_to_weight(fitgpt_df) # 55-65 kg and 80-90kg there are people who has lowest fat percentage - needs further analysing
    # frequency_to_experience(fitgpt_df) # More hours you train, more experience you gain
    # weight_bmi(fitgpt_df) # they are dependant, so that one of the parameters can be removed
    # duration_calories(fitgpt_df) # they are dependant, the more hours you train, the more calories you burn
    # BPM_by_Age(fitgpt_df) # there is no big difference in BPM based on the age of the client
    # gender_workout_type(fitgpt_df) # strange thing is that the number of males on Yoga classes is much bigger than the number of females
    # fat_experience_level(fitgpt_df)


    # knn(fitgpt_df)
    log_reg(fitgpt_df)

if __name__ == '__main__':
    main()