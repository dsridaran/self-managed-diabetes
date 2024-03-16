using CSV, DataFrames, Statistics, Random
using Impute: KNN

# Function to read data
function load_data(filepath)
    CSV.read(filepath, DataFrame)
end

# Function to split data into training and testing sets
function split_data(df, treatment_var, outcomes_var, seed = 123; train_proportion = 0.5)
    Random.seed!(seed)
    n = nrow(df)
    idx = shuffle(1:n)
    train_idx = idx[1:floor(Int, train_proportion * n)]
    test_idx = idx[(floor(Int, train_proportion * n) + 1):end]

    train_X = df[train_idx, Not([treatment_var, outcomes_var])]
    test_X = df[test_idx, Not([treatment_var, outcomes_var])]
    train_treatments = df[train_idx, treatment_var]
    test_treatments = df[test_idx, treatment_var]
    train_outcomes = df[train_idx, outcomes_var]
    test_outcomes = df[test_idx, outcomes_var]

    ((train_X, train_treatments, train_outcomes), (test_X, test_treatments, test_outcomes))
end

# Function for cleaning and imputation
function preprocess_data(train_X, test_X; threshold = 0.25)
    
    # Identify and remove variables with high missingness
    var_desc = describe(train_X, :nmissing, :nnonmissing)
    var_desc.propmissing = var_desc.nmissing ./ (var_desc.nmissing .+ var_desc.nnonmissing)
    vars_to_remove = filter(row -> row.propmissing > threshold, var_desc).variable

    select!(train_X, Not(vars_to_remove))
    select!(test_X, Not(vars_to_remove))

    # Impute missing values using KNN
    lnr = IAI.OptKNNImputationLearner(random_seed = 15095)
    train_X = IAI.fit_transform!(lnr, train_X)
    test_X = IAI.transform(lnr, test_X)

    # Create train flags for ease of identification
    train_X[!, :train_flag] .= 1
    test_X[!, :train_flag] .= 0

    (train_X, test_X)
end

# Main function for data imputation
function main()
    
    df = load_data("data/input/diabetes_patients.csv")
    treatment_var = "diabetes_course"
    outcomes_var = "phys_health_status"

    (train, test) = split_data(df, treatment_var, outcomes_var)
    (train_X, test_X) = preprocess_data(train[1], test[1])
    
    # Re-attach treatment and outcomes
    train_X[:, treatment_var] = train[2]
    train_X[:, outcomes_var] = train[3]
    test_X[:, treatment_var] = test[2]
    test_X[:, outcomes_var] = test[3]

    # Combine processed train and test data
    df = vcat(train_X, test_X)

    # Create binary variable for un-safe household
    conditions = [
        (df.depressed_household .== "yes") .| (df.alcohol_household .== "yes") .| (df.drugs_household .== "yes") .| (df.prison_household .== "yes"),
        (df.depressed_household .== "no") .& (df.alcohol_household .== "no") .& (df.drugs_household .== "no") .& (df.prison_household .== "no")
    ]
    values = ["yes", "no"]
    df.unsafe_household .= ifelse.(conditions[1], values[1], ifelse.(conditions[2], values[2], ""))

    # Create condition count variables
    df[!, :physical_conditions] = (df.ever_any_cancer .== "yes") + (df.cvd .== "yes") + (df.chd .== "yes") + (df.stroke .== "yes") + (df.ever_asthma .== "yes") + (df.copd .== "yes") + (df.arthritis .== "yes") + (df.kidney_disease .== "yes") + (df.deaf .== "yes") + (df.blind .== "yes")
    
    df[!, :mental_conditions] = (df.depression .== "yes") + (df.cognitive_decline .== "yes")
    df[!, :any_conditions] = df.mental_conditions .+ df.physical_conditions
    df[!, :activities_daily_living] = (df.mobility .== "yes") + (df.dressing_bathing .== "yes") + (df.errands .== "yes") + (df.concentration .== "yes");
     
    # Save the processed data
    CSV.write("data/input/imputed_diabetes_patients.csv", df)
end

# Run the main function
main()