import numpy as np
import pandas as pd


""" 
Made by Laura van Geene, Spring 2024, for MSc thesis research.

Python file to be run in combination with full_model_greenlight.py.

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


from full_model_greenlight import import_figaro, price_model_energy, DNB_substitution

def do_SPA(powers, A_trans, delta_v):

    results_SPA = {} #dictionary for results
    #Calculate SPA
    for power in powers:
        result = (A_trans.values ** power) @ delta_v
        result.index = multi_index #multi-index to be able to interpret results
        results_SPA[f'tier_{power}'] = result #add to dictionary

    #add zeroth tier
    results_SPA['tier_0']=delta_v

    #find contributions for Electricity and Other scientific activities (the two most affected sectors)
    contributions_SPA={}

    # loop through each tier in the results_SPA dictionary
    for tier, df in results_SPA.items():
        energy_row=df.loc["the Netherlands", :].loc["Electricity, gas, steam and air conditioning supply", :]
        science_row=df.loc["the Netherlands", :].loc["Other professional, scientific and veterinary activities", :]
        
        contributions_SPA[tier] = {
            'energy': energy_row.values,
            'science_row': science_row.values,
            }

    return results_SPA, contributions_SPA

#%% Data import

filepath="data/Figaro/"
filepath2="data/"

#import all IO data
A, A_old, L, A_trans, L_trans, inv_diag_x_, x, Y, v0, fco2, multi_index, start_NL, end_NL, v_df, sector_labels = import_figaro(filepath)

#import elasticities
Bun_elasticities=pd.read_excel(f'{filepath2}Elasticities Thesis.xlsx', sheet_name='BUN', usecols=[2], index_col=None)
Bun_elasticities=Bun_elasticities.squeeze() #turns into array
Bun_elas_list=Bun_elasticities.tolist() #turns into list

#%% IO analysis

""" Do input-output analysis """

#define price
phi=0.15

#run price model for Dutch carbon tax
new_price_energy, delta_p1, v_tax, delta_v_energy, tax_value = price_model_energy(L_trans, v0, fco2, phi, start_NL, end_NL, multi_index)

#define sectors and elasticities
sectors = sector_labels
sigma_list = Bun_elas_list

#run analysis for base elasticity
A_star_sub_base=DNB_substitution(sectors, sigma_list, delta_v_energy, delta_p1, multi_index, A_old, v0, start_NL, end_NL, None, None)


""" Run model for different elasticities"""

#defines sectors and elasticities that need to be changed
weird_sector= ["Electricity, gas, steam and air conditioning supply"]
the_weird_sigmas=[0.3, 0.5, 0.7, 1.0]

A_star_sub_results={}

for weird_sigma in the_weird_sigmas:
    A_star_sub=DNB_substitution(sectors, sigma_list, delta_v_energy, delta_p1, multi_index, A_old, v0, start_NL, end_NL, weird_sector, weird_sigma)
    A_star_sub_results[f'{weird_sigma}']=A_star_sub

#extract different A_stars
A_star_sub_3=A_star_sub_results['0.3']
A_star_sub_5=A_star_sub_results['0.5']
A_star_sub_7=A_star_sub_results['0.7']
A_star_sub_one=A_star_sub_results['1.0']

#%% Structural Path Analysis

#Do SPA for different A-matrices

the_As=[A_old, A_star_sub_base, A_star_sub_3, A_star_sub_5, A_star_sub_7, A_star_sub_one]
the_markers=[0, 0.16, 0.3, 0.5, 0.7, 1.0]

collect_results_SPA={}
collect_contributions_SPA={}
powers=np.arange(1, 10, 1)

for A_matrix, marker in zip(the_As, the_markers):

    A_trans=np.transpose(A_matrix)
    results_SPA, contributions_SPA=do_SPA(powers, A_trans, delta_v_energy)
    collect_results_SPA[f'{marker}']=results_SPA
    collect_contributions_SPA[f'{marker}']=contributions_SPA


print("I have finished the SPA :)")

#Optional export to Excel

# #%%
# #use ExcelWriter
# with pd.ExcelWriter('results_SPA.xlsx') as writer:
#     #Go over each dictionary in the collect_results 
#     for dict_key, dict_value in collect_results_SPA.items():
#         #go over each dataframe in the different dictionaries
#         for df_key, df_value in dict_value.items():
#             #save the different dataframes in a different excel sheet --> only two sectors that we are interested in
#             df_value.loc[[("the Netherlands", "Electricity, gas, steam and air conditioning supply"), ("the Netherlands","Other professional, scientific and veterinary activities")]].to_excel(writer, sheet_name=f"{dict_key}_{df_key}", index=True)

# print("I have exported the results of the SPA to Excel :)")
