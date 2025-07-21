## Final_Dissertation_Code
  
London faces deep social disparities, especially in housing. Low-income individuals are often excluded from both private rentals and homeownership. Gentrification intensifies this issue, as rising housing costs displace long-term residents in once-affordable areas. 
  
This project explores whether patterns in housing or architectural design, as reflected in residential planning applications, can help predict neighborhood gentrification. 
  
The first stage uses `Natural Language Processing` (`NLP`) with transformer models (`Bertopic` structure) to extract features from free-text planning application descriptions provided by the **Greater London Authority** (**GLA**). 
  
The second stage predicts gentrification levels using `GCN`, `LSTM`, or `Transformer` (**feature suggestion**) models, combining extracted features with external datasets (**census data**) across **LSOAs**.