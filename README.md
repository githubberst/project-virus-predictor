# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 4 - Tackling the West Nile Virus outbreak with data

### Problem Statement

As a newly appointed member of the Disease And Treatment Agency's division of Societal Cures In Epidemiology and New Creative Engineering (DATA-SCIENCE), our task is to develop an efficient plan for the deployment of pesticides in response to the endemicity of West Nile Virus in the city. With the establishment of a surveillance and control system by the Department of Public Health, there is an opportunity to leverage collected data on mosquito populations to derive valuable insights. 

The aim is to strategically allocate resources and minimize costs associated with pesticide use while ensuring public health and safety. Our expertise in data analysis and modeling will be instrumental in formulating an effective pesticide deployment plan to combat the West Nile Virus outbreak in the Windy City.

### Objectives

* The primary objective entails constructing a robust predictive model to facilitate informed decision-making by the city of Chicago regarding the strategic allocation of pesticide spraying for mosquito control.

* Another component of the project involves conducting a comprehensive cost-benefit analysis. This analysis encompasses projecting direct and indirect costs associated with pesticide coverage and evaluating the corresponding benefits yielded by pesticide application.

Throughout the project, we will undertake crucial steps such as preprocessing the data, conducting Exploratory Data Analysis (EDA) and feature engineering. Finally, the performance of our models will be evaluated based on the highest ROC AUC score, ensuring the effectiveness of our model in predicting WNV presence at the traps correctly. 


---

### Datasets

<b> Primary Data </b>:

The training dataset comprises data from the years 2007, 2009, 2011, and 2013, while the test dataset comprises data from the years 2008, 2010, 2012, and 2014.
To facilitate data organization, mosquito count records exceeding 50 are split into separate entries, ensuring that the number of mosquitoes does not exceed this limit.

<b> Weather Data </b>:

The weather dataset from NOAA contains weather conditions recorded between 2007 and 2014, specifically during the months of the tests. It is believed that hot and dry conditions are more conducive to West Nile virus than cold and wet conditions. 
The weather data is available for two stations: 
1. CHICAGO O'HARE INTERNATIONAL AIRPORT (Latitude: 41.995, Longitude: -87.933, Elevation: 662 ft. above sea level)
2. CHICAGO MIDWAY INTL ARPT (Latitude: 41.786, Longitude: -87.752, Elevation: 612 ft. above sea level)

<b> Spray Data </b>:

The City of Chicago conducts mosquito spraying efforts, and GIS data for their spraying activities in 2011 and 2013 is provided. Spraying can reduce mosquito populations and potentially eliminate the presence of West Nile virus.

<b> Map Data </b>:

The map files, mapdata_copyright_openstreetmap_contributors.rds and mapdata_copyright_openstreetmap_contributors.txt, are sourced from OpenStreetMap and are primarily intended for use in visualizations.

Acknowledgements:
This competition is sponsored by the Robert Wood Johnson Foundation. Data is provided by the Chicago Department of Public Health.
https://www.kaggle.com/competitions/predict-west-nile-virus

### Data Dictionary
The data dictionary for the four datasets utilized in this project is provided below for reference.

`train_df`

Period: 2007, 2009, 2011, and 2013

|Feature|Type|Description|
|:---|:---:|:---|
|<b>Date</b>|*object*|Date that the WNV test is performed|
|<b>Address</b>|*object*|Approximate address of the location of trap, this is used to send to the GeoCoder|
|<b>Species</b>|*object*|The species of mosquitos|
|<b>Block</b>| *int64*|Block number of address|
|<b>Street</b>|*object*|Street name|
|<b>Trap</b>|*object*|Id of the trap|
|<b>AddressNumberAndStreet</b>|*object*|Approximate address returned from GeoCoder|
|<b>Latitude</b>|*float64*|Latitude returned from GeoCoder|
|<b>Longitude</b>|*float64*|Longitude returned from GeoCoder|
|<b>AddressAccuracy</b>|*int64*|Accuracy returned from GeoCoder|
|<b>NumMosquitos</b>|*int64*|Number of mosquitoes caught in this trap|
|<b>WnvPresent</b>|*int64*|Whether West Nile Virus was present in these mosquitos. 1 means WNV is present, and 0 means not present. |

<br>

`test_df`

Period: 2008, 2010, 2012, and 2014

|Feature|Type|Description|
|:---|:---:|:---|
|<b>Id</b>|*int64*|The id of the record|
|<b>Date</b>|*object*|Date that the WNV test is performed|
|<b>Address</b>|*object*|Approximate address of the location of trap, this is used to send to the GeoCoder|
|<b>Species</b>|*object*|The species of mosquitos|
|<b>Block</b>| *int64*|Block number of address|
|<b>Street</b>|*object*|Street name|
|<b>Trap</b>|*object*|Id of the trap|
|<b>AddressNumberAndStreet</b>|*object*|Approximate address returned from GeoCoder|
|<b>Latitude</b>|*float64*|Latitude returned from GeoCoder|
|<b>Longitude</b>|*float64*|Longitude returned from GeoCoder|
|<b>AddressAccuracy</b>|*int64*|Accuracy returned from GeoCoder|

<br>

`weather_df`

Period: 2007, 2008, 2009, 2010, 2011, 2012, 2013, and 2014

|Feature|Type|Description|
|:---|:---:|:---|
|<b>Date</b>|*object*|Date of record|
|<b>Station</b>|*int64*|Station number, either 1 or 2|
|<b>Tmax</b>|*int64*|Maximum temperature in Degrees Fahrenheit|
|<b>Tmin</b>|*int64*|Minimum temperature in Degrees Fahrenheit|
|<b>Tavg</b>|*object*|Average temperature in Degrees Fahrenheit|
|<b>Depart</b>| *object*|Temperature departure from normal in Degrees Fahrenheit|
|<b>DewPoint</b>|*int64*|Average Dew Point in Degrees Fahrenheit|
|<b>WetBulb</b>|*object*|Average Wet Bulb in Degrees Fahrenheit|
|<b>Heat</b>|*object*|Absolute temperature difference of Tavg from base temperature of 65 Degrees Fahrenheit if Tavg < 65|
|<b>Cool</b>|*object*|Absolute temperature difference of Tavg from base temperature of 65 Degrees Fahrenheit if Tavg > 65|
|<b>Sunrise</b>|*object*|Time of Sunrise (Calculated, not observed)|
|<b>Sunset</b>|*object*|Time of Sunset (Calculated, not observed)|
|<b>CodeSum</b>|*object*|Weather Phenomena, refer to CodeSum Legend below|
|<b>Depth</b>|*object*|Snow / ice in inches|
|<b>Water1</b>|*object*|Water equivalent of Depth|
|<b>SnowFall</b>| *object*|Snowfall in inches and tenths|
|<b>PrecipTotal</b>|*object*|Rainfall and melted snow in inches and hundredths|
|<b>StnPressure</b>|*object*|Average station pressure in inches of HG|
|<b>SeaLevel</b>|*object*|Average sea level pressure in inches of HG|
|<b>ResultSpeed</b>|*float64*|Resultant wind speed in miles per hour|
|<b>ResultDir</b>|*int64*|Resultant wind direction in Degrees|
|<b>AvgSpeed</b>|*object*|Average wind speed in miles per hour|

<br>

<b>CodeSum Legend</b>

|code| explanation| 
|:-|:-|
|+FC| TORNADO/WATERSPOUT|
|FC | FUNNEL CLOUD|
|TS | THUNDERSTORM|
|GR | HAIL|
|RA | RAIN|
|DZ | DRIZZLE|
|SN | SNOW|
|SG | SNOW GRAINS|
|GS | SMALL HAIL &/OR SNOW PELLETS|
|PL | ICE PELLETS|
|IC | ICE CRYSTALS|
|FG+ | HEAVY FOG (FG & LE.25 MILES VISIBILITY)|
|FG | FOG|
|BR | MIST|
|UP | UNKNOWN PRECIPITATION|
|HZ | HAZE|
|FU | SMOKE|
|VA | VOLCANIC ASH|
|DU | WIDESPREAD DUST|
|DS | DUSTSTORM|
|PO | SAND/DUST WHIRLS|
|SA | SAND|
|SS | SANDSTORM|
|PY | SPRAY|
|SQ | SQUALL|
|DR | LOW DRIFTING|
|SH | SHOWER|
|FZ | FREEZING|
|MI | SHALLOW|
|PR | PARTIAL|
|BC | PATCHES|
|BL | BLOWING|
|VC | VICINITY|
|- | LIGHT|
|+ | HEAVY|
|"NO SIGN" | MODERATE|

`spray_df`

Period: 2011, and 2013

|Feature|Type|Description|
|:---|:---:|:---|
|<b>Date</b>|*object*|Date of the spray|
|<b>Time</b>|*object*|Time of the spray|
|<b>Latitude</b>|*float64*|Latitude returned from GeoCoder|
|<b>Longitude</b>|*float64*|Longitude returned from GeoCoder|

<br>

---

### Conclusion

In summary:
We can predict areas of importance to prioritise efforts on spray deployment. 
- The agency should consider the following collectively to determine priority areas within Chicago: 
    - `Month`, specifically in the months of Aug as we have seen in our EDA. 
    - a higher `roll_mean_28_Tavg`, as a higher 28day mean temperature in the summer months are optimal for mosquito gestation and activity, which would increase likelihood of WNV+ trap locations in those with `species` of `Culex Pipiens`.  
    - the `sunset`, `sunrise` and `timediff` (i.e. length of night) also prove to be important in predicting WNV+ as it is likely that `Culex Pipiens` favour around 11-12 hours of night time for their 'munching' activities. That said, the optimal length of night from a relatively late sunrise and sunset also tends to occur in Aug.  
    - as `trap` is feature engineered indirectly with `WnvPresent` and the same trap is usually deployed at the same locations, it is no surprise that this is an important feature. This also means that the model is dependent on how `trap` ID is assigned currently.

- Next, we also observed that spray effectiveness is somewhat weak. Through use of our model's predictions, we would be able to augment spray deployments and optimise its use effectively. 

Upon revisiting the problem statement, we are reminded of our two primary objectives:

1) <b>Objective 1</b>

    - Our first objective was to construct a robust predictive model to facilitate informed decision-making by the city of Chicago regarding the strategic allocation of pesticide spraying for mosquito control.
    - With a test ROC AUC of 0.88 and kaggle ROC AUC of 0.76, our model can efficiently predict and classify 76-88% of WNV+ traps correctly. The model can be deployed for early operations planning and decision making on which areas to be sprayed during the summer months. This will help to focus vector control efforts on potential hotspots, and prevent rise in unnecessary cases and deaths, and unnecessary use of financial and manpower resources to deploy the sprays. Nevertheless, the model would still need to be updated with up-to-date data so that it can continue to generate accurate predictions for continued future use.

2) <b>Objective 2</b>

    - Our second objective was to conduct a comprehensive cost-benefit analysis. 
    - Based on a 2013-year scenario cost benefit analysis, we estimate that our model approach outperforms the current approach with a benefit-cost ratio of 353. (if required, please refer to the pdf deck for details) 

---

### Recommendations


Instead of the current approach (i.e. as of 2015 when the Kaggle competition was launched by CDPH) where the agency seems to deploy adulticide spraying based on positive WNV traps<sup>1</sup>, we urge the agency to consider a data driven approach using our model to predict priority areas ahead of traps turning WNV positive. This also means that instead of spraying after the event where traps turn WNV positive, this model allows the agency to take pre-emptive action before the area is filled with WNV positive mosquitoes, which could have already led to incidence of human WNV cases. From a public health point of view, this would allow us to hammer down on the growth in WNV presence in the area, before the actual spike in human cases. 

Since the model is able to predict probabilities of WNV presence, the agency is able to accord different risk levels to these 77 community areas in Chicago, e.g. RED for highest risk, AMBER for medium risk, GREEN for low risk. Courses of action to take for each risk level can be drawn up, with RED being assigned the highest response measures (e.g. adulticiding + larvaciding + public education/comms campaign). Such information can also be made online for businesses and public to plan ahead. 


<sup>1. https://www.chicago.gov/city/en/depts/cdph/supp_info/infectious/west_nile_virus_surveillancereports.html</sup>



### Next steps

As we continue collecting data, we should update and refine our model to improve its predictive power and adapt to changes in mosquito behavior and climate conditions.
We also recommend further studies to better understand association with urban landscape features, that may seem to impact WNV transmission, and include it into the model.
By keeping a keen eye on mosquito species, their activity, and related geographical clusters, we can remain proactive in our fight against the West Nile Virus.





