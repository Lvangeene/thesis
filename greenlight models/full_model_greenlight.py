import numpy as np
import pandas as pd
import cbsodata  #API by CBS to load different dataset



""" 
Made by Laura van Geene, Spring 2024, for MSc thesis research.

Code requires data files to run:
- Figaro IO-table (figaro_ixi_2015.csv)
- Figaro CO2 extensions (CO2_footprints_2015.csv)
- Figaro sector and region labels (region_labels.txt and sector_labels.txt)
- List of elasticities (for this research: Elasticities Thesis.xlsx)
- List of expenditures to include in analysis from CBS Dataset (main_expenditures.xlsx)
- Installation of Python library cbsodata (can be installed via pip)
- Concordance table for CBS and Figaro data (Concordance Table COICOP-Figaro.csv)

Put datafiles in folder called 'data'

"""


""" ALL DATA IMPORTS """

#function to import all the Figaro data and split it into the proper matrices/vectors
def import_figaro(filepath):

    #import data
    data = pd.read_csv(f'{filepath}figaro_ixi_2015.csv', sep=',')
    #data not a square matrix, need to filter for Y and V
    F_co2=pd.read_csv(f'{filepath}CO2_footprints_2015.csv', sep=';', header=[0], dtype={'time_peiod': int, 'obs_value': float, 'decimals': int, 'unit_mult': int})
    
    #final demand starts at column index 2945
    #exclude the final six rows with value-added
    Y=data.iloc[:2944, 2945:]
    
    #value added is in the last six rows of the data set
    #exclude the columns with final demand
    V=data.iloc[-6:, :2945]
    V=V.drop(columns=["rowLabels"])
    V=V.sum(axis=0) #to get total final demand
    V=V.values
    V=V.reshape((2944,1))

    #select the Z-matrix in between the columns and rows with final demand and value-added
    Z=data.iloc[:2944, :2945]
    Z=Z.drop(columns=["rowLabels"])

    #import sector and region labels from official Figaro legend
    sectors=pd.read_csv(f'{filepath}sector_labels.txt', sep='\t', index_col=[0], header=None) 
    regions=pd.read_csv(f'{filepath}region_labels.txt', sep='\t', index_col=[0], header=None)
        
    #make sure that they have the right format
    sector_labels = sectors.reset_index(drop=False)
    sector_labels=sector_labels.squeeze()
    region_labels=regions.reset_index(drop=False)
    region_labels=region_labels.squeeze()
        
    ## Create a multi-index to assign to the IO_variables when needed
    multi_index = pd.MultiIndex.from_product([region_labels, sector_labels], names=['region', 'sector'])

    #format the CO2-data
    Fco2=F_co2.drop(['time_period', 'decimals', 'unit_measure', 'unit_mult', 'obs_status', 'conf_status', 'last_update'], axis=1)
    Fco2=Fco2.iloc[:-47, :] #drop the household emissions

    #pivoting to get a matrix format
    pivot_Fco2 = Fco2.pivot_table(index=['ref_area', 'industry'], columns= ['counterpart_area', 'sto'], values='obs_value', aggfunc='first')
    pivot_Fco2=pivot_Fco2.sum(axis=1) #sum all final demand categories
    F_co2=pivot_Fco2.values #drop all the indexes
    F_co2 =F_co2.reshape((2944,1))

    #calculate A, L, A_trans, L_trans
    Z_sum=Z.sum(axis=1)
    Y_sum=Y.sum(axis=1)
    Z_sum=Z.sum(axis=1)
    x_out=Z_sum + Y_sum
    x_out.values #total output to calculate A

    #calculate inv diag x
    x_ = x_out.copy()
    x_[x_!=0] = 1/x_[x_!=0]
    inv_diag_x_ = np.diag(x_)

    A= Z@inv_diag_x_
    I = np.identity(A.shape[0])
    L = np.linalg.inv(I-A)
    A_trans = np.transpose(A)
    L_trans=np.linalg.inv(I - A_trans)

    #recalculate x, v, fco2, and Y
    x= L @ Y_sum
    v0=inv_diag_x_ @ V
    fco2=inv_diag_x_@F_co2

    Y=Y
    Y_reg = Y.groupby(level=0, axis=1, sort=False).sum()
    Y = Y_reg.sum(axis=1)

    r = region_labels.shape[0]
    s = sector_labels.shape[0]
    
    #NL is thet 43th country in the list of countries
    start_NL=42*s
    end_NL=start_NL+s

    v_df=pd.DataFrame(v0, index=multi_index, columns=["Value-added"])
    #required for one of the scenarios

    A_old=A.copy()
    return A, A_old, L, A_trans, L_trans, inv_diag_x_, x, Y, v0, fco2, multi_index, start_NL, end_NL, v_df, sector_labels

#function to import the CBS consumption data
def CBS_Data(filepath2):

    ''' Function requires specification of expenditures that you want to filter from the large dataset '''

    # Downloading table list
    toc = pd.DataFrame(cbsodata.get_table_list())

    # Downloading entire dataset (can take up to 30s)
    data = pd.DataFrame(cbsodata.get_data('83679NED')) #83679NED is the ID of the dataset

    #Downloading metadata (if necessary for later research)
    metadata = pd.DataFrame(cbsodata.get_meta('83679NED', 'DataProperties')) #83679NED is the ID of the dataset
    #print(metadata[['Key','Title']])

#Code above directly copied from CBS website: https://www.cbs.nl/en-gb/our-services/open-data/statline-as-open-data/quick-start-guide

    #change Perioden (Periods) to numeric for filtering
    data["Perioden"] = pd.to_numeric(data["Perioden"])

    #filter for the right year
    data_2015= data.loc[data["Perioden"] == 2015 ]

    #drop Perioden column & drop ID columns, since no longer nececssary
    data_2015=data_2015.drop(["Perioden", "ID"], axis=1)
    
    #filter for only the Huishoud kenmerken about age ('jaar')
    data_age=data_2015[data_2015["KenmerkenHuishoudens"].str.contains('jaar')]

    #remove the most aggregated categories
    categories_to_keep= [
    'Hoofdkostwinner: tot 25 jaar', 'Hoofdkostwinner: 25 tot 35 jaar',
    'Hoofdkostwinner: 35 tot 45 jaar', 'Hoofdkostwinner: 45 tot 55 jaar',
    'Hoofdkostwinner: 55 tot 65 jaar', 'Hoofdkostwinner: 65 tot 75 jaar',	
    'Hoofdkostwinner: 75 jaar of ouder']

    #filters for the categories
    data_age=data_age[data_age['KenmerkenHuishoudens'].isin(categories_to_keep)]

    #rename the columns
    data_age=data_age.rename(columns={'Bestedingscategorieen': 'Expenditure Category', 'KenmerkenHuishoudens': 'Age Group', 
    'Bestedingsaandeel_1': 'Percentage of Expenditure'})

    #fill missing values with 0
    data_age=data_age.fillna(0)

    #import list with expenditure categories that you want to keep
    filtered_exp=pd.read_excel(f'{filepath2}main_expenditures.xlsx', index_col=[0], header=None)

    #THIS RESEARCH: Main categories of consumption available in dataset
    #(except: narcotics, due to complexity of allocating in the bridge matrix)

    #changes to list to use as filter
    filtered_exp= filtered_exp.reset_index(drop=False)
    filtered_exp= filtered_exp.squeeze()
    
    #subset the large dataset for only the expenditures of interest
    subset_main=data_age[data_age["Expenditure Category"].isin(filtered_exp)]
  
    #pivot table to have proper format
    consumption_table=subset_main.pivot(index="Expenditure Category", columns = 'Age Group', values= 'Percentage of Expenditure')
    
    return consumption_table


""" All IO-FUNCTIONS """

#Conventional Leontief price model with carbon tax for all sectors in the Netherlands
#Requires specification of carbon tax (phi)

def price_model_NL(L_trans, v0, fco2, phi, start_NL, end_NL, multi_index):
      
    #make tax rate variable
    tax_rate = np.copy(fco2)
    tax_rate.fill(phi)

    #makes sure that carbon tax is only applied to the Netherlands
    for i in range(len(tax_rate)):
        if i < start_NL or i > end_NL:
            tax_rate[i] = 0

    #calculate tax value 
    tax_value=tax_rate*fco2 
    
    #calculate new value-added 
    v_tax = np.add(v0, tax_value)
        
    #new price with new value-added
    new_price=L_trans@v_tax
    delta_p1=L_trans@(tax_value)
        
    delta_v=pd.DataFrame(tax_value, index=multi_index, columns=["Change to value-added"])   

    return new_price, delta_p1, v_tax, delta_v, tax_value

#Conventional Leontief price model with carbon tax for Dutch Energy sector
def price_model_energy(L_trans, v0, fco2, phi, start_NL, end_NL, multi_index):
      
    #make tax rate variable
    tax_rate = np.copy(fco2)
    tax_rate.fill(phi)

    #makes sure that carbon tax is only applied to the Netherlands
    for i in range(len(tax_rate)):
        if i == 2711: #2711 is the index of the electricity sector
            tax_rate[i] = phi
        else:
            tax_rate[i] = 0
        
    #calculate tax value 
    tax_value=tax_rate*fco2 
    
    #calculate new value-added 
    v_tax = np.add(v0, tax_value)
        
    #new price with new value-added
    new_price=L_trans@v_tax
    delta_p1=L_trans@(tax_value)
        
    delta_v=pd.DataFrame(tax_value, index=multi_index, columns=["Change to value-added"])   

    return new_price, delta_p1, v_tax, delta_v, tax_value

""" Recalculating the A-matrix """

#function to calculate new technical coefficients following the price change
def DNB_substitution(sectors, sigma_list, delta_v, delta_p1, multi_index, A_old, v0, start_NL, end_NL, weird_sector=None, weird_sigma=None):
    row_indexes = []
    delta_a_targets = []
    A_star_sub=A_old.copy()

    #for loop to replace the calculate new technical coefficients
    for sector, sigma in zip(sectors, sigma_list): 
        #sector that needs to be changed
        target_index= ("the Netherlands", sector)
        #find row index of sector that needs to be changed
        row_index = delta_v.index.get_loc(target_index)
    
        old_coeff_targets=A_old.iloc[row_index, start_NL:end_NL]
        #print(old_coeff_targets)
        #use row_index to find relative price change for that sector
        p1_change_target = delta_p1[row_index]
                
        #determine the capital labor inputs using the row_index --> value-added inputs
        z_c_target = v0[row_index]

        #new list to hold the updated coefficients for each sector
        new_coeff_list = []
        
        for index, coeff in enumerate(old_coeff_targets):
            #print(coeff)
        #if there are sectors which require a different sigma, fill in different sigma
            if weird_sector is not None and sector in weird_sector:
                sigma = weird_sigma
            else: 
                sigma = sigma
                            
            #calculate the new technical coefficient for the target sector
            influence_elas = 1 + sigma * p1_change_target #subsitution effect
                    
            if sector == "Activities of extraterritorial organisations and bodies": 
                #IO-data is zero for this sector --> replace with 1 to prevent errors
                bottom = 1.0
                top = 1.0
            else:
                top = coeff + z_c_target
                bottom = coeff + (z_c_target * influence_elas)
            
            fraction = top / bottom
  

            #new technical coefficient
            new_A_coeff = coeff * fraction

            
            # Append the new coefficient to the new_coeff_list
            new_coeff_list.append(float(new_A_coeff))

        #print("New list", new_coeff_list)
        
        #place values into A_star matrix
        A_star_sub.iloc[row_index, start_NL:end_NL]=new_coeff_list
      
    return A_star_sub

#function uses adapted A to rerun the Leontief price model

def calc_L_and_p(A_star, v_tax):
    #calculate new L
    I = np.identity(A_star.shape[0])
    L=np.linalg.inv(I - A_star)

    #calculate new price
    p_tax= np.transpose(L)@v_tax

    return p_tax, L
 
#function collects results into a dataframe

def collect_results(new_p, p_standard_tax, multi_index):

    #Collect these results
    data_tax = {'Column1': new_p.flatten(), 'Column2': p_standard_tax.flatten()}
    column_label_tax=pd.Index(["no substitution", "Capital/labour substitution"])

    #put them in a dataframe
    results_tax=pd.DataFrame(data=data_tax, index=multi_index)
    results_tax.columns=column_label_tax
    
    return results_tax

def do_input_output_NL(filepath, phi, sectors, sigma_list, weird_sector, weird_sigma):

    #import input-output data
    A, A_old, L, A_trans, L_trans, inv_diag_x_, x, Y, v0, fco2, multi_index, start_NL, end_NL, v_df, sector_labels = import_figaro(filepath)

    #run price model for Dutch carbon tax
    new_price, delta_p1, v_tax, delta_v, tax_value = price_model_NL(L_trans, v0, fco2, phi, start_NL, end_NL, multi_index)
    
    #capital-labour substitution
    A_star_sub = DNB_substitution(sectors, sigma_list, delta_v, delta_p1, multi_index, A_old, v0, start_NL, end_NL, weird_sector, weird_sigma)

    #recalculate price change and Leontief Inverse
    p_DNB_tax, L_DNB=calc_L_and_p(A_star_sub, v_tax)
    
    #collect results into dataframe
    results_tax=collect_results(new_price, p_DNB_tax, multi_index)
        
    return results_tax, p_DNB_tax, new_price, A_star_sub, delta_v

def do_input_output_energy(filepath, phi, sectors, sigma_list, weird_sector, weird_sigma):

    #import input-output data
    A, A_old, L, A_trans, L_trans, inv_diag_x_, x, Y, v0, fco2, multi_index, start_NL, end_NL, v_df, sector_labels = import_figaro(filepath)

    #run price model for Dutch carbon tax
    new_price_energy, delta_p1, v_tax, delta_v_energy, tax_value = price_model_energy(L_trans, v0, fco2, phi, start_NL, end_NL, multi_index)
    
    #capital-labour substitution
    A_star_sub_energy = DNB_substitution(sectors, sigma_list, delta_v_energy, delta_p1, multi_index, A_old, v0, start_NL, end_NL, weird_sector, weird_sigma)

    #recalculate price change and Leontief Inverse
    p_DNB_energy, L_DNB_energy=calc_L_and_p(A_star_sub_energy, v_tax)
    
    #collect results into dataframe
    results_tax_energy=collect_results(new_price_energy, p_DNB_energy, multi_index)
        
    return results_tax_energy, p_DNB_energy, new_price_energy, A_star_sub_energy, delta_v_energy


""" DATA MANIPULATIONS"""

#function applies concordance matrix to input-output outcomes
def concordance(price_change_LPM, filepath2, start_NL, end_NL):
    
    price_change_NL=price_change_LPM[start_NL:end_NL]

    #import concordance table
    conc_table=pd.read_csv(f'{filepath2}Concordance Table COICOP-Figaro.csv', sep=';', index_col=[0], header=[0])
    
    #exclude TOTAL and NOTES columns
    conc_table=conc_table.iloc[:, :-2]
    
    #transpose price change and concordance table to allow multiplication
    vector_transposed = price_change_NL.transpose()
    concordance_transposed = conc_table.transpose()

    #multiply the rows of the price change vector (sectors) with their corresponding weight in concordance table
    collapsed_price_change = (vector_transposed @ concordance_transposed.values).transpose()

    return collapsed_price_change


""" CONSUMER PRICE INDEX CALCULATIONS """

#function calculation the change in CPI following price change & substitution
def CPI_Change(collapsed_price_change, consumption_table):
    
    #in LPM, base price is always one
    ones=np.ones_like(collapsed_price_change)

    #old consumption    
    OLD=np.transpose(ones)@consumption_table
    
    #new consumption
    NEW=np.transpose(collapsed_price_change)@consumption_table   
    
    #change in Consumer price chanfge
    delta_CPI=((NEW-OLD)/OLD)*100

    return delta_CPI

#calculate CPI and make df with results
def calc_CPI(p1_collapsed, consumption_table):

    #calculate CPI
    CPI=CPI_Change(p1_collapsed, consumption_table) 

    return CPI

#calculate and collect the results into a nice dataframe
def collect_CPIs(CPI_LPM, CPI_sub):
    #compile the results into dataframes
    df_CPI=pd.concat([CPI_LPM, CPI_sub])
        
    #make them pretty
    df_CPI=df_CPI.set_index([['without substitution', 'with capital/labour subst.']])
    
    return df_CPI


# Calculates and collects CPI for Dutch and global carbon tax
def do_CPI(new_price_NL, p_DNB_tax, start_NL, end_NL, filepath2, consumption_table):

    # Apply concordance and calculate CPI of LPM price change
    collapsed_LPM = concordance(new_price_NL, filepath2, start_NL, end_NL)
    CPI_LPM=calc_CPI(collapsed_LPM, consumption_table)    
        
    # Apply concordance and calculate CPI of capital/labour price change
    collapsed_sub = concordance(p_DNB_tax, filepath2, start_NL, end_NL)
    CPI_sub= calc_CPI(collapsed_sub, consumption_table) # Calculate CPI

    # Compile results of CPI for Dutch and global carbon tax
    results_CPI = collect_CPIs(CPI_LPM, CPI_sub)

    # Rename the columns to English
    results_CPI= results_CPI.rename(columns={"Hoofdkostwinner: 25 tot 35 jaar": "25 to 35", "Hoofdkostwinner: 35 tot 45 jaar": "35 to 45", "Hoofdkostwinner: 45 tot 55 jaar": '45 to 55', "Hoofdkostwinner: 55 tot 65 jaar": "55 to 65", "Hoofdkostwinner: 65 tot 75 jaar": "65 to 75", "Hoofdkostwinner: 75 jaar of ouder": "75 or older", "Hoofdkostwinner: tot 25 jaar": "up to 25" })
    
    # Move columns with youngest age group to proper position
    young_col = results_CPI.pop('up to 25')
    results_CPI.insert(0, 'up to 25', young_col, allow_duplicates=True)

    return results_CPI

""" 

SUMMARIZE ALL INTO ONE FUNCTION 

"""

# Do analysis for Dutch carbon tax
def do_analysis(phi, sectors, sigma_list, filepath, filepath2, consumption_table, start_NL, end_NL, weird_sector=None, weird_sigma=None):

    # Run input-output model with Dutch carbon tax
    results_tax_NL, p_DNB_tax, new_price, A_star_sub, delta_v_NL = do_input_output_NL(filepath, phi, sectors, sigma_list, weird_sector, weird_sigma)

    # Calculates the CPI for a Dutch and energy tax
    results_NL= do_CPI(new_price, p_DNB_tax, start_NL, end_NL, filepath2, consumption_table)    
    
    return results_NL, results_tax_NL, A_star_sub

#Do analysis for energy tax
def do_analysis_energy(phi, sectors, sigma_list, filepath, filepath2, consumption_table, start_NL, end_NL, weird_sector=None, weird_sigma=None):

    # Run input-output model with energy tax
    results_tax_energy, p_DNB_energy, new_price_energy, A_star_sub_energy, delta_v_energyy = do_input_output_energy(filepath, phi, sectors, sigma_list, weird_sector, weird_sigma)

    # Calculates the CPI for a Dutch and energy tax
    results_energy= do_CPI(new_price_energy, p_DNB_energy, start_NL, end_NL, filepath2, consumption_table)    
    
    return results_energy, results_tax_energy, A_star_sub_energy


""" In case you want to do the analysis until the concordance only"""

def do_consumption_analysis(filepath, filepath2, phi, sectors, sigma_list, start_NL, end_NL, weird_sector, weird_sigma):
    # Run input-output model with Dutch carbon tax
    results_tax_NL, p_DNB_tax, new_price, A_star_sub, delta_v_NL = do_input_output_NL(filepath, phi, sectors, sigma_list, weird_sector=None, weird_sigma=None)
    
    #apply concordance and calculate CPI of LPM price change
    collapsed_LPM_NL=concordance(new_price, filepath2, start_NL, end_NL)
        
    return collapsed_LPM_NL

def do_consumption_analysis_energy(filepath, filepath2, phi, sectors, sigma_list, start_NL, end_NL, weird_sector=None, weird_sigma=None):
    # Run input-output model with energy tax
    results_tax_energy, p_DNB_energy, new_price_energy, A_star_sub_energy, delta_v_energyy = do_input_output_energy(filepath, phi, sectors, sigma_list, None, None)
    
    #apply concordance and calculate CPI of LPM price change
    collapsed_LPM_energy=concordance(new_price_energy, filepath2, start_NL, end_NL)
        
    return collapsed_LPM_energy


"""
Analysis for a hypothetical global tax 

Note: in the current model, input-substitution only happens in the Dutch economy, due to lack of data 
for global substitution elasticities. Hence, the outcomes of the flexible IO model for a global tax were 
excluded from the results of this thesis, since it is not reasonable to assume that input-substitution
only happens in the Dutch economy.

"""


def price_model_world(L_trans, v0, fco2, phi, start_NL, end_NL, multi_index):
    
    #make tax rate variable
    tax_rate = np.copy(fco2)
    tax_rate.fill(phi)

    #calculate tax value
    tax_value_world=tax_rate*fco2
    
    #calculate new value-added vector
    v_tax_world= np.add(v0, tax_value_world)
    
    #new price with new value-added
    p1_all_world=L_trans@v_tax_world

    #calculate tax value 
    tax_value=tax_rate*fco2 
    
    #calculate new value-added 
    v_tax = np.add(v0, tax_value)
        
    #new price with new value-added
    new_price_world=L_trans@v_tax
    delta_p1_world=L_trans@(tax_value)
        
    delta_v_world=pd.DataFrame(tax_value, index=multi_index, columns=["Change to value-added"])   

    return new_price_world, delta_p1_world, delta_v_world, v_tax_world

#global carbon tax
def do_input_output_world(filepath, phi, sectors, sigma_list, weird_sector=None, weird_sigma=None):

    #import input-output data
    A, A_old, L, A_trans, L_trans, inv_diag_x_, x, Y, v0, fco2, multi_index, start_NL, end_NL, v_df, sector_labels = import_figaro(filepath)

    #run price model for Dutch carbon tax
    new_price_world, delta_p1_world, delta_v_world, v_tax_world = price_model_world(L_trans, v0, fco2, phi, start_NL, end_NL, multi_index)
    
    #capital-labour substitution
    A_star_DNB = DNB_substitution(sectors, sigma_list, delta_v_world, delta_p1_world, multi_index, A_old, v0, start_NL, end_NL, None, None)

    #recalculate price change and Leontief Inverse
    p_DNB_tax_world, L_DNB=calc_L_and_p(A_star_DNB, v_tax_world)
    
    #collect results into dataframe
    results_tax_world=collect_results(new_price_world, p_DNB_tax_world, multi_index)
    
    return results_tax_world, p_DNB_tax_world, new_price_world, A_star_DNB, delta_v_world

#same elasticities for all sectors
def do_analysis_world(phi, sectors, default_sigma, filepath, filepath2, consumption_table, start_NL, end_NL):

    #run input-output model with global carbon tax
    results_LPM_world, p_DNB_tax_world, new_price_world, A_star_DNB_world, delta_v_world=do_input_output_world(filepath, phi, sectors, default_sigma, None, None)

    #calculates the CPI for a Dutch and global carbon tax
    results_world=do_CPI(new_price_world, p_DNB_tax_world, start_NL, end_NL, filepath2, consumption_table)    
    
    return results_world, results_LPM_world, A_star_DNB_world, delta_v_world



#%% Run analysis

#set file paths
filepath="data/Figaro/"
filepath2="data/"

#imports IO data and calculates MRIO variables
A, A_old, L, A_trans, L_trans, inv_diag_x_, x, Y, v0, fco2, multi_index, start_NL, end_NL, v_df, sector_labels = import_figaro(filepath)

#import the consumption table from CBS
consumption_table=CBS_Data(filepath2)

#Import DNB Elasticities
Bun_elasticities=pd.read_excel(f'{filepath2}Elasticities Thesis.xlsx', sheet_name='BUN', usecols=[2], index_col=None)
Bun_elasticities=Bun_elasticities.squeeze() #turns into array
Bun_elas_list=Bun_elasticities.tolist() #turns into list for later use in model

#%% Dutch tax

""" Results Dutch Tax"""

#set parameters
sectors= sector_labels
the_prices=[0.05, 0.15, 0.25]
sigma_list=Bun_elas_list

#make dictionaries to save results
results_CPI_NL={} #CPI outcomes
results_LPM_NL={} #sectoral price changes
results_A_star_sub_NL={} #changed A-matrix
consumption_NL={} #consumption categories


#loop to run analysis
for price in the_prices:
    #calculate CPI, sectoral price changes, and adapted A-matrix
    result_NL, result_tax_NL, A_star_sub = do_analysis(price, sectors, sigma_list, filepath, filepath2, consumption_table, start_NL, end_NL)
    results_CPI_NL[f'{price}']=result_NL
    results_LPM_NL[f'{price}']= result_tax_NL
    results_A_star_sub_NL[f'{price}']=A_star_sub
    
    #consumption analysis
    collapsed_LPM_NL=do_consumption_analysis(filepath, filepath2, price, sectors, sigma_list, start_NL, end_NL, None, None)
    consumption_NL[f'{price}']=collapsed_LPM_NL


#Consumer Price Index
results_CPI_50_NL=results_CPI_NL['0.05']
results_CPI_150_NL=results_CPI_NL['0.15']
results_CPI_250_NL=results_CPI_NL['0.25']

#Sectoral price changes
results_50_LPM_NL=results_LPM_NL['0.05']
results_150_LPM_NL=results_LPM_NL['0.15']
results_250_LPM_NL=results_LPM_NL['0.25']

#collect consumption analysis results
consumption_50=consumption_NL['0.05']
consumption_50_NL=pd.Series(consumption_50.squeeze())
consumption_150=consumption_NL['0.15']
consumption_150_NL=pd.Series(consumption_150.squeeze())
consumption_250=consumption_NL['0.25']
consumption_250_NL=pd.Series(consumption_250.squeeze())

#%% Energy tax

""" Results Energy Tax"""

sectors= sector_labels

#make dictionaries to save results
results_CPI_energy={} #CPI outcomes
results_LPM_energy={} #sectoral price changes
results_A_star_sub_energy={} #adapted A-matrix
consumption_energy={} #consumption categories

#loop to run analysis
for price in the_prices:
    #calculate CPI, sectoral price changes, adapted A-matrix
    result_energy, result_tax_energy, A_star_sub_energy = do_analysis_energy(price, sectors, sigma_list, filepath, filepath2, consumption_table, start_NL, end_NL, None, None)
    results_CPI_energy[f'{price}']=result_energy
    results_LPM_energy[f'{price}']= result_tax_energy
    results_A_star_sub_energy[f'{price}']=A_star_sub_energy

    #consumption analysis
    collapsed_LPM_energy=do_consumption_analysis_energy(filepath, filepath2, price, sectors, sigma_list, start_NL, end_NL, None, None)
    consumption_energy[f'{price}']=collapsed_LPM_energy

#consumer price index
results_CPI_50_energy=results_CPI_energy['0.05']
results_CPI_150_energy=results_CPI_energy['0.15']
results_CPI_250_energy=results_CPI_energy['0.25']

#sectoral price changes
results_50_LPM_energy=results_LPM_energy['0.05']
results_150_LPM_energy=results_LPM_energy['0.15']
results_250_LPM_energy=results_LPM_energy['0.25']

#collect results consumption analysis
consumption_50=consumption_energy['0.05']
consumption_50_energy=pd.Series(consumption_50.squeeze())
consumption_150=consumption_energy['0.15']
consumption_150_energy=pd.Series(consumption_150.squeeze())
consumption_250=consumption_energy['0.25']
consumption_250_energy=pd.Series(consumption_250.squeeze())

#%% Global Tax

""" Results Global Tax"""

#set up dictionaries to save results
results_CPI_global={} #CPI outcomes
results_LPM_global={} #sectoral price changes


#loop to run analysis
for price in the_prices:
    #calculate CPI, sectoral price changes, adapted A_matrix
    result_global, result_tax_global, A_star_sub_global, delta_v_world = do_analysis_world(price, sectors, sigma_list, filepath, filepath2, consumption_table, start_NL, end_NL)
    results_CPI_global[f'{price}']=result_global
    results_LPM_global[f'{price}']= result_tax_global

#consumer price index
results_CPI_50_global=results_CPI_global['0.05']
results_CPI_150_global=results_CPI_global['0.15']
results_CPI_250_global=results_CPI_global['0.25']

#sectoral price changes
results_50_LPM_global=results_LPM_global['0.05']
results_150_LPM_global=results_LPM_global['0.15']
results_250_LPM_global=results_LPM_global['0.25']