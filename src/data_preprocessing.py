import pandas as pd
import numpy as np

def read_and_select_columns(file_path, columns):
    """Load data and select required columns."""
    data = pd.read_csv(file_path, usecols = columns)
    return data

def preprocess_patients(data):
    """Process patient data to include only relevant information and categories."""

    # Remove patients with unknown treatment or outcomes
    data = data[(data['DIABEDU'] <= 2.0)]
    data = data[data['_PHYS14D'] != 9.0]
    data = data[data['_MENT14D'] != 9.0]

    # Process outcome response variables
    rename_columns = {
        '_PHYS14D': 'phys_health_status',
        '_MENT14D': 'ment_health_status',
        'PHYSHLTH': 'phys_health_days',
        'MENTHLTH': 'ment_health_days'
    }
    data.rename(columns = rename_columns, inplace = True)
    data['phys_health_days'].replace(88, 0)
    data['ment_health_days'].replace(88, 0)
    patients = data
    
    # Personal doctor
    conditions = [
        patients['PERSDOC2'] == 1.0,
        patients['PERSDOC2'] == 2.0,
        patients['PERSDOC2'] == 3.0
    ]
    
    values = ['only_one', 'more_than_one', 'no']
    
    patients = patients.assign(personal_doctor = np.select(conditions, values, default = ''))
    patients.drop('PERSDOC2', axis = 1, inplace = True)

    # Diabetes education
    conditions = [
        patients['DIABEDU'] == 1.0,
        patients['DIABEDU'] == 2.0
    ]
    
    values = ['yes', 'no']
    
    patients = patients.assign(diabetes_course = np.select(conditions, values, default = ''))
    patients.drop('DIABEDU', axis = 1, inplace = True)

    # Age
    patients.rename(columns = {'_AGE80': 'age'}, inplace = True)

    # Gender
    conditions = [
        patients['SEXVAR'] == 1.0,
        patients['SEXVAR'] == 2.0
    ]
    
    values = ['male', 'female']
    
    patients = patients.assign(sex = np.select(conditions, values, default = ''))
    patients.drop('SEXVAR', axis = 1, inplace = True)

    # Language
    conditions = [
        patients['QSTLANG'] == 1.0,
        patients['QSTLANG'] == 2.0
    ]
    
    values = ['english', 'spanish']
    
    patients = patients.assign(language = np.select(conditions, values, default = ''))
    patients.drop('QSTLANG', axis = 1, inplace = True)

    # Race
    conditions = [
        patients['_IMPRACE'] == 1.0,
        patients['_IMPRACE'] == 2.0,
        patients['_IMPRACE'] == 3.0,
        patients['_IMPRACE'] == 4.0,
        patients['_IMPRACE'] == 5.0,
        patients['_IMPRACE'] == 6.0
    ]
    
    values = ['white', 'black', 'asian', 'native', 'hispanic', 'other']
    
    patients = patients.assign(race = np.select(conditions, values, default = ''))
    patients.drop('_IMPRACE', axis = 1, inplace = True)

    # Veteran
    conditions = [
        patients['VETERAN3'] == 1.0,
        patients['VETERAN3'] == 2.0,
        patients['VETERAN3'] == 7.0,
        patients['VETERAN3'] == 9.0
    ]
    
    values = ['yes', 'no', 'unknown', 'unknown']
    
    patients = patients.assign(veteran = np.select(conditions, values, default = ''))
    patients.drop('VETERAN3', axis = 1, inplace = True)

    # BMI
    conditions = [
        patients['_BMI5CAT'] == 1.0,
        patients['_BMI5CAT'] == 2.0,
        patients['_BMI5CAT'] == 3.0,
        patients['_BMI5CAT'] == 4.0
    ]
    
    values = ['underweight', 'normal', 'overweight', 'obese']
    
    patients = patients.assign(bmi = np.select(conditions, values, default = ''))
    patients.drop('_BMI5CAT', axis = 1, inplace = True)

    # Metropolitan status
    conditions = [
        patients['_METSTAT'] == 1.0,
        patients['_METSTAT'] == 2.0
    ]
    
    values = ['yes', 'no']
    
    patients = patients.assign(metro = np.select(conditions, values, default = ''))
    patients.drop('_METSTAT', axis = 1, inplace = True)

    conditions = [
        patients['MSCODE'] == 1.0,
        patients['MSCODE'] == 2.0,
        patients['MSCODE'] == 3.0,
        patients['MSCODE'] == 5.0
    ]
    
    values = ['city_center', 'city', 'county', 'outside']
    
    patients = patients.assign(metro_granular = np.select(conditions, values, default = ''))
    patients.drop('MSCODE', axis = 1, inplace = True)

    # Education
    conditions = [
        patients['EDUCA'] == 1.0,
        patients['EDUCA'] == 2.0,
        patients['EDUCA'] == 3.0,
        patients['EDUCA'] == 4.0,
        patients['EDUCA'] == 5.0,
        patients['EDUCA'] == 6.0
    ]
    
    values = ['no_high_school', 'no_high_school', 'high_school_some', 'high_school_graduate', 'college_some', 'college_graduate']
    
    patients = patients.assign(education = np.select(conditions, values, default = ''))
    patients.drop('EDUCA', axis = 1, inplace = True)

    # Employment
    conditions = [
        patients['EMPLOY1'] == 1.0,
        patients['EMPLOY1'] == 2.0,
        patients['EMPLOY1'] == 3.0,
        patients['EMPLOY1'] == 4.0,
        patients['EMPLOY1'] == 5.0,
        patients['EMPLOY1'] == 6.0,
        patients['EMPLOY1'] == 7.0,
        patients['EMPLOY1'] == 8.0
    ]
    
    values = ['employed', 'employed', 'unemployed', 'unemployed', 'non_labor_force', 'non_labor_force', 'non_labor_force', 'non_labor_force']
    
    patients = patients.assign(employment = np.select(conditions, values, default = ''))
    patients.drop('EMPLOY1', axis = 1, inplace = True)

    # Income
    conditions = [
        patients['INCOME2'] == 1.0,
        patients['INCOME2'] == 2.0,
        patients['INCOME2'] == 3.0,
        patients['INCOME2'] == 4.0,
        patients['INCOME2'] == 5.0,
        patients['INCOME2'] == 6.0,
        patients['INCOME2'] == 7.0,
        patients['INCOME2'] == 8.0
    ]
    
    values = ['poverty', 'poverty', 'low', 'low', 'low', 'middle', 'middle', 'high']
    
    patients = patients.assign(income = np.select(conditions, values, default = ''))
    patients.drop('INCOME2', axis = 1, inplace = True)

    # Partners
    conditions = [
        patients['MARITAL'] == 1.0,
        patients['MARITAL'] == 2.0,
        patients['MARITAL'] == 3.0,
        patients['MARITAL'] == 4.0,
        patients['MARITAL'] == 5.0,
        patients['MARITAL'] == 6.0
    ]
    
    values = ['couple', 'single', 'single', 'single', 'single', 'couple']
    
    patients = patients.assign(partner = np.select(conditions, values, default = ''))
    patients.drop('MARITAL', axis = 1, inplace = True)

    # Children
    conditions = [
        patients['CHILDREN'] == 88.0,
        patients['CHILDREN'] == 1.0,
        patients['CHILDREN'] == 2.0,
        (patients['CHILDREN'] >= 3.0) & (patients['CHILDREN'] <= 4.0),
        (patients['CHILDREN'] >= 5.0) & (patients['CHILDREN'] <= 87.0)
    ]
    
    values = ['none', 'one', 'two', 'three_to_four', 'five_plus']
    
    patients = patients.assign(children = np.select(conditions, values, default = ''))
    patients.drop('CHILDREN', axis = 1, inplace = True)

    # Household members
    conditions = [
        patients['ACEDEPRS'] == 1.0,
        patients['ACEDEPRS'] == 2.0,
        patients['ACEDEPRS'].isna()
    ]
    
    values = ['yes', 'no', 'no']
    
    patients = patients.assign(depressed_household = np.select(conditions, values, default = ''))
    patients.drop('ACEDEPRS', axis = 1, inplace = True)

    conditions = [
        patients['ACEDRINK'] == 1.0,
        patients['ACEDRINK'] == 2.0,
        patients['ACEDRINK'].isna()
    ]
    
    values = ['yes', 'no', 'no']
    
    patients = patients.assign(alcohol_household = np.select(conditions, values, default = ''))
    patients.drop('ACEDRINK', axis = 1, inplace = True)

    conditions = [
        patients['ACEDRUGS'] == 1.0,
        patients['ACEDRUGS'] == 2.0,
        patients['ACEDRUGS'].isna()
    ]
    
    values = ['yes', 'no', 'no']
    
    patients = patients.assign(drugs_household = np.select(conditions, values, default = ''))
    patients.drop('ACEDRUGS', axis = 1, inplace = True)

    conditions = [
        patients['ACEPRISN'] == 1.0,
        patients['ACEPRISN'] == 2.0,
        patients['ACEPRISN'].isna()
    ]
    
    values = ['yes', 'no', 'no']
    
    patients = patients.assign(prison_household = np.select(conditions, values, default = ''))
    patients.drop('ACEPRISN', axis = 1, inplace = True)

    # Exercise
    conditions = [
        patients['EXERANY2'] == 1.0,
        patients['EXERANY2'] == 2.0
    ]
    
    values = ['yes', 'no']
    
    patients = patients.assign(exercise_past_month = np.select(conditions, values, default = ''))
    patients.drop('EXERANY2', axis = 1, inplace = True)

    # Sleep
    conditions = [
        patients['SLEPTIM1'] < 5.0,
        (patients['SLEPTIM1'] >= 5.0) & (patients['SLEPTIM1'] < 7.0),
        (patients['SLEPTIM1'] >= 7.0) & (patients['SLEPTIM1'] < 10.0),
        (patients['SLEPTIM1'] >= 10.0) & (patients['SLEPTIM1'] < 25.0),
    ]
    
    values = ['very_low', 'low', 'healthy', 'very_high']
    
    patients = patients.assign(sleep = np.select(conditions, values, default = ''))
    patients.drop('SLEPTIM1', axis = 1, inplace = True)

    # Smoking
    conditions = [
        patients['SMOKE100'] == 1.0,
        patients['SMOKE100'] == 2.0
    ]
    
    values = ['yes', 'no']
    
    patients = patients.assign(ever_smoked_100 = np.select(conditions, values, default = ''))
    patients.drop('SMOKE100', axis = 1, inplace = True)

    conditions = [
        patients['SMOKDAY2'] == 1.0,
        patients['SMOKDAY2'] == 2.0,
        patients['SMOKDAY2'] == 3.0,
        patients['SMOKDAY2'].isna()
    ]
    
    values = ['often', 'sometimes', 'none', 'none']
    
    patients = patients.assign(current_smoker = np.select(conditions, values, default = ''))
    patients.drop('SMOKDAY2', axis = 1, inplace = True)

    # E-smoking
    conditions = [
        patients['ECIGARET'] == 1.0,
        patients['ECIGARET'] == 2.0,
        patients['ECIGARET'].isna()
    ]
    
    values = ['yes', 'no', 'no']
    
    patients = patients.assign(ever_e_smoked = np.select(conditions, values, default = ''))
    patients.drop('ECIGARET', axis = 1, inplace = True)

    conditions = [
        patients['ECIGNOW'] == 1.0,
        patients['ECIGNOW'] == 2.0,
        patients['ECIGNOW'] == 3.0,
        patients['ECIGNOW'].isna()
    ]
    
    values = ['often', 'sometimes', 'none', 'none']
    
    patients = patients.assign(current_e_smoker = np.select(conditions, values, default = ''))
    patients.drop('ECIGNOW', axis = 1, inplace = True)

    # Drinking
    patients.loc[patients['AVEDRNK3'].isna(), 'AVEDRNK3'] = 0
    patients.loc[patients['AVEDRNK3'] == 99.0, 'AVEDRNK3'] = np.nan
    patients.loc[patients['AVEDRNK3'] == 77.0, 'AVEDRNK3'] = np.nan
    patients.rename(columns = {'AVEDRNK3': 'average_alcohol_month'}, inplace = True)

    # Health conditions
    o1 = ['CHCOCNCR', 'CHCSCNCR', 'CVDINFR4', 'CVDCRHD4', 'CVDSTRK3', 'ASTHMA3', 'CHCCOPD2', 'HAVARTH4', 'ADDEPEV3', 'CHCKDNY2']
    o2 = ['DEAF', 'BLIND', 'DECIDE', 'DIFFWALK', 'DIFFDRES', 'DIFFALON']
    n1 = ['ever_any_cancer', 'ever_skin_cancer', 'cvd', 'chd', 'stroke', 'ever_asthma', 'copd', 'arthritis', 'depression', 'kidney_disease']
    n2 = ['deaf', 'blind', 'concentration', 'mobility', 'dressing_bathing', 'errands']
    old_vars = o1 + o2
    new_vars = n1 + n2

    for i in range(16):
        
        old_var = old_vars[i]
        new_var = new_vars[i]
        
        conditions = [
            patients[old_var] == 1.0,
            patients[old_var] == 2.0
        ]
        
        values = ["yes", "no"]
        
        patients = patients.assign(temp = np.select(conditions, values, default = ''))
        patients.drop(old_var, axis = 1, inplace = True)
        patients.rename(columns = {'temp': new_var}, inplace = True)

    # Asthma
    conditions = [
        patients['ASTHNOW'] == 1.0,
        patients['ASTHNOW'] == 2.0,
        patients['ASTHNOW'].isna()
    ]
    
    values = ["yes", "no", "no"]
    
    patients = patients.assign(current_asthma = np.select(conditions, values, default = ''))
    patients.drop('ASTHNOW', axis = 1, inplace = True)

    # Diabetes
    conditions = [
        patients['DIABETE4'] == 1.0,
        patients['DIABETE4'] == 2.0,
        patients['DIABETE4'] == 3.0,
        patients['DIABETE4'] == 4.0
    ]
    
    values = ["yes", "no", "no", "borderline"]
    
    patients = patients.assign(diabetes = np.select(conditions, values, default = ''))
    patients.drop('DIABETE4', axis = 1, inplace = True)

    # Cognitive decline
    conditions = [
        patients['CIMEMLOS'] == 1.0,
        patients['CIMEMLOS'] == 2.0,
        patients['CIMEMLOS'].isna()
    ]
    
    values = ["yes", "no", "no"]
    
    patients = patients.assign(cognitive_decline = np.select(conditions, values, default = ''))
    patients.drop('CIMEMLOS', axis = 1, inplace = True)

    # Pregnancy
    conditions = [
        patients['PREGNANT'] == 1.0,
        patients['PREGNANT'] == 2.0,
        patients['PREGNANT'].isna()
    ]
    
    values = ["yes", "no", "no"]
    
    patients = patients.assign(pregnant = np.select(conditions, values, default = ''))
    patients.drop('PREGNANT', axis = 1, inplace = True)

    # Cancer count
    conditions = [
        patients['CNCRDIFF'] == 1.0,
        patients['CNCRDIFF'] == 2.0,
        patients['CNCRDIFF'] == 3.0,
        patients['CNCRDIFF'].isna()
    ]
    
    values = ["one", "two", "three", "zero"]
    
    patients = patients.assign(number_cancers = np.select(conditions, values, default = ''))
    patients.drop('CNCRDIFF', axis = 1, inplace = True)

    # Cancer type
    conditions = [
        patients['CNCRTYP1'].isin([1.0]),
        patients['CNCRTYP1'].isin([2.0, 3.0, 4.0]),
        patients['CNCRTYP1'].isin([10.0]),
        patients['CNCRTYP1'].isin([19.0, 20.0]),
        patients['CNCRTYP1'].isin([21.0, 22.0]),
        patients['CNCRTYP1'].isin([77.0, 99.0]),
        patients['CNCRTYP1'].isin([np.nan])
    ]
    
    values = ["breast", "cerv_endo_ovar", "colon", "prostate", "skin", "", "none"]
    
    patients = patients.assign(cancer_type = np.select(conditions, values, default = "other"))
    patients.drop('CNCRTYP1', axis = 1, inplace = True)

    # Health coverage
    conditions = [
        patients['age'] >= 65,
        patients['_HCVU651'] == 1.0,
        patients['_HCVU651'] == 2.0,
    ]
    
    values = ["over_65", "yes", "no"]
    
    patients = patients.assign(health_coverage = np.select(conditions, values, default = ''))
    patients.drop('_HCVU651', axis = 1, inplace = True)

    return data

def clean_data(file_path, output_file_path):

    # Define columns to be read from file
    required_columns =[
        # Response variables (health outcomes)
        '_PHYS14D', '_MENT14D', 'PHYSHLTH', 'MENTHLTH',
        # Demographics
        '_AGE80', 'SEXVAR', 'QSTLANG', '_IMPRACE', 'VETERAN3', '_BMI5CAT', 'PREGNANT',
        # Socio-economics
         '_METSTAT', 'MSCODE', 'EDUCA', 'EMPLOY1', 'INCOME2', 
        # Relationships
        'MARITAL', 'CHILDREN', 'ACEDEPRS', 'ACEDRINK', 'ACEDRUGS', 'ACEPRISN',
        # Lifestyle
        'EXERANY2', 'SLEPTIM1', 'SMOKE100', 'SMOKDAY2', 'AVEDRNK3', 'ECIGARET', 'ECIGNOW',
        # Health conditions
        'CVDINFR4', 'CVDCRHD4', 'CVDSTRK3', 'ASTHMA3', 'ASTHNOW', 'CHCCOPD2', 'HAVARTH4', 'ADDEPEV3', 'CHCKDNY2', 'DIABETE4', 'CIMEMLOS',
        # Activities of daily living
        'DECIDE', 'DIFFWALK', 'DIFFDRES', 'DIFFALON',
        # Disabilities
        'DEAF', 'BLIND',
        # Health system usage
        '_HCVU651',
        # Cancer history
        'CHCSCNCR', 'CHCOCNCR', 'CNCRDIFF', 'CNCRTYP1', 
        # Treatment variables
        'DIABEDU', 'PERSDOC2'
    ]
    
    data = read_and_select_columns(file_path, required_columns)
    processed_data = preprocess_patients(data)
    processed_data.to_csv(output_file_path, index = False)

if __name__ == "__main__":
    clean_data()