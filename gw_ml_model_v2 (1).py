# Function to calculate NVDI scores and average biomass
def gw_model_v2( farmid, measure_date ):
    import pyodbc
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Polygon, MultiPolygon
    from datetime import datetime

    from gw_logger import get_gw_logger
    logger = get_gw_logger()
    
    logger.info(f'Start gw_model_v2("{farmid}", "{measure_date}")')
    time_start = datetime.now()

    try:
        #Fetch data from Paddocktrac(SQL server DB) and group them into sets of 50 each
        server = 'bob.cares.missouri.edu'
        database = 'GrazingWedge_Production'
        username = 'Grazingwedge'
        password = 'GW_Prod24'

        connection_string = (
            f'DRIVER={{SQL Server}};'
            f'SERVER={server};'
            f'DATABASE={database};'
            f'UID={username};'
            f'PWD={password};'
        )

        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()

        # Execute your SQL query
        cursor.execute("""
                
        with GroupedPoints as

            (SELECT 
                    app.*, 
                    ((tare - rawdistance) * 0.0859536) AS height,
                        pad.paddock, pad.shape.STAsText() as paddock_coordinates,
                        fm.farm
                FROM 
                    dbo.appupload app 
                    INNER JOIN dbo.Paddocks pad 
                    ON app.paddockid = pad.id 
                    AND app.farmid = pad.farmid
                    INNER JOIN dbo.farm fm  
                    ON pad.farmid = fm.id

                WHERE
                    app.farmid = """ + farmid + """ AND 
                    timestamp = '""" + measure_date + """'),

         weather as
        
                (
                     SELECT 
                    farm_id,-- max([date]) as timestamp,
                    ((AVG((temperature_max_F + temperature_min_F)/2) - 32) * (5/9)) AS Avg_21D_Temp_C,
                    SUM(ETo_in) AS Cum_21D_ETo_in,
                    SUM(precipitation_in) AS Cum_21D_Precip_in,
                    SUM(GDD) AS Cum_21D_GDD,
                    AVG(Available_Soil_Water_in) AS Avg_21D_SWB_in,
                    AVG(Available_Soil_Water_percent) AS Avg_21D_SWB_pct
                FROM dbo.FarmWeatherData
                
                WHERE farm_id = """ + farmid + """
                  AND [date] > DATEADD(DAY, -23, '""" + measure_date + """')
                  AND [date] < DATEADD(DAY, -3, '""" + measure_date + """')
                  
                GROUP BY farm_id
                
           
                ),
                
         hgt as (     
                SELECT 
                        farmid,farm,paddockid,paddock,paddock_coordinates,timestamp,lat,lng,
                        AVG(height) AS mean_height
                        --,count(*) as count
                    FROM 
                        GroupedPoints
        
                    GROUP BY 
                        farmid,farm,paddockid,paddock,paddock_coordinates,timestamp,lat,lng
          --          ORDER BY farmid,paddockid,paddock,paddock_coordinates,timestamp
         )
        
          
         select hgt.*, weather.*, 
         round((-26.05 + (8.68 * Avg_21D_Temp_C) + (0.021 * power(Avg_21D_Temp_C , 2)) - 
         (0.0071 * power(Avg_21D_Temp_C , 3)) )/100 , 2) as Growth_rt_21D,
         round((Avg_21D_SWB_pct *  round((-26.05 + (8.68 * Avg_21D_Temp_C) + (0.021 * power(Avg_21D_Temp_C , 2))
         - (0.0071 * power(Avg_21D_Temp_C , 3)) )/100 , 2)) ,2) as GR_SWB_21D
        
         from hgt
         left join weather
         on weather.farm_id = hgt.farmid
         ORDER BY farmid,paddockid,paddock,paddock_coordinates,timestamp

        """)

        # --------------------
        # Fetch the column names
        columns = [column[0] for column in cursor.description]

        # Fetch the data into a list of tuples
        results = cursor.fetchall()

        # Convert rows to list of tuples (unpacking each row)
        data = [tuple(row) for row in results]

        # Check the shape of the fetched data
        logger.info(f"Number of rows fetched: {len(data)}")
        logger.info(f"Number of columns in each row: {len(data[0]) if data else 'No data'}")

        # Create a Pandas DataFrame from the results
        df = pd.DataFrame(data, columns=columns)

        #forming a polygon around the center point of each set
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add new columns for Date and Coordinates
        df['Date'] = df['timestamp'].apply(lambda date: date.strftime('%Y-%m-%dT23:59:59Z'))

        # Create polygons with a length of 4m and store list-of-lists coordinates directly
        offset = 0.000011
        coordinates = []

        for i in range(len(df) - 1):
            lat1, lng1 = df.loc[i, ['lat', 'lng']]
            lat2, lng2 = df.loc[i + 1, ['lat', 'lng']]

            poly = Polygon([
                (lng1, lat1 + offset),
                (lng1, lat1 - offset),
                (lng2, lat2 - offset),
                (lng2, lat2 + offset),
                (lng1, lat1 + offset)
            ])

            coords_list = [[x, y] for x, y in poly.exterior.coords]
            coordinates.append(coords_list)

        # Pad last row with None to match DataFrame length
        coordinates.append(None)

        # Assign directly to the 'Coordinates' column
        df['Coordinates'] = coordinates

        # Optional: display full coordinate lists
        pd.set_option('display.max_colwidth', None)

        # Print the updated DataFrame
        #print(df)
        # Close the cursor and connection
        cursor.close()
        conn.close()
        # df.to_csv('Mutligeojson.csv', index=False)

        # ---------- Sentinel HUb Authentication and Collection Id ----------
        logger.info("Start connection to Sentinel Hub")
        
        from ipyleaflet import Map, GeoData, basemaps, LayersControl
        import matplotlib.pyplot as plt
        import numpy as np
        import getpass
        import json
        from gw_sentinel_subscription import planet_settings_v2

        # Sentinel Hub Python Library
        from sentinelhub import (
            CRS,
            DataCollection,
            Geometry,
            SentinelHubStatistical,
            SentinelHubStatisticalDownloadClient,
            SHConfig,
            parse_time
        )

        # Planet Visualization Library
        # from planet_style import set_theme
        # set_theme()

        logger.info("Packages Imported")
        print("Packages Imported")

        # Import the SHConfig class from sentinelhub-py
        from sentinelhub import SHConfig

        # Create a configuration object
        config = SHConfig()
        
        # Get planet subscription info - needs CLIENT_ID, CLIENT_SECRET, COLLECTION_ID
        client_settings = planet_settings_v2()

        # Check if the client ID and client secret are provided
        status = ''
        if client_settings and client_settings['CLIENT_ID'] and client_settings['CLIENT_SECRET']:
            config.sh_client_id = client_settings['CLIENT_ID']
            config.sh_client_secret = client_settings['CLIENT_SECRET']
            
            status = "Connected to Sentinel Hub"
        else:
            status = "No credentials found, please provide the OAuth client ID and secret."
        logger.info(status)    
        print(status)

        collection_id = client_settings['COLLECTION_ID']
        PlanetScope_data_collection = DataCollection.define_byoc(collection_id)

        status = f"Using Sentinel Hub Collection '{collection_id}'"
        logger.info(status)
        print(status)

        # --------- Calculating VIs -----------
        from sentinelhub import SentinelHubStatistical, SentinelHubStatisticalDownloadClient, Geometry, CRS, parse_time
        from shapely.geometry import Polygon
        from datetime import datetime, timedelta
        import ast
        from shapely import wkt

        data=df


        #Below are methods to convert coordinates into polygon if its not in expected format
        # Convert subplot_coordinates from string to geometry
        # data['coord'] = data['coord'].apply(lambda x: Polygon(eval(x)))
        data['Coordinates'] = data['Coordinates'].apply(lambda x: Polygon(eval(str(x))))
        # data['Coordinates'] = data['Coordinates'].apply(lambda x: Polygon(ast.literal_eval(str(x))))
        # data['Coordinates'] = data['Coordinates'].apply(lambda x: x.to_wkt() if isinstance(x, Polygon) else x)
        # data['Coordinates'] = data['Coordinates']

        # df['DT'] = '2022-09-16'
        # data['Img_date'] = df['DT']
        from datetime import timedelta

        # Convert 'Date' column to datetime if it's not already
        data['Date'] = pd.to_datetime(data['Date'])

        # Subtract one day and format as 'YYYY-MM-DD'
        data['Img_date'] = (data['Date']).dt.strftime('%Y-%m-%d')

        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(data, geometry='Coordinates')
        gdf.crs = "EPSG:4326"  # Assuming the coordinates are in WGS84

        # Convert date column to datetime
        gdf['Img_date'] = pd.to_datetime(gdf['Img_date'])

        # Define the data collection
        input_data = SentinelHubStatistical.input_data(data_collection=PlanetScope_data_collection)

        # Define a resolution, below is the minimum pixel size of planet image
        resx = 0.00001
        resy = 0.00001

        # Evaluation script to calculate NDVI, GNDVI, EVI, SAVI, and MSAVI
        evalscript = """
        //VERSION=3

        function setup() {
            return {
            input: [
                {
                bands: [
                    "red",
                    "green",
                    "blue",
                    "nir",
                    "rededge",
                    "dataMask",
                    "clear"
                ]
                }
            ],
            output: [
            {
                id: "ndvi",
                bands: 1
            },
            {
                id: "gndvi",
                bands: 1
            },
            {
                id: "evi",
                bands: 1
            },
            {
                id: "savi",
                bands: 1
            },
            {
                id: "msavi",
                bands: 1
            },
            {
                id: "ndre",
                bands: 1
            },
            {
                id: "Clre",
                bands: 1
            },
            {
                id: "SRre",
                bands: 1
            },
            {
                id: "red",
                bands: 1
            },
            {
                id: "green",
                bands: 1
            },
            {
                id: "blue",
                bands: 1
            },
            {
                id: "nir",
                bands: 1
            },
            {
                id: "rededge",
                bands: 1
            },
                {
                id: "dataMask",
                bands: 1
                }
            ]
            }
        }

        function isClear(clear) {
            return clear === 1;
        }

        function evaluatePixel(samples) {
            let ndvi = (samples.nir - samples.red) / (samples.nir + samples.red);
            let gndvi = (samples.nir - samples.green) / (samples.nir + samples.green);
            let evi = 2.5 * (samples.nir - samples.red) / (samples.nir + 6.0 * samples.red - 7.5 * samples.blue + (1.0*10000));
            let L = 0.5;
            let savi = (samples.nir - samples.red) * (1 + L) / (samples.nir + samples.red + (L*10000));
            let s = 10000; 
            let msavi = (2 * (samples.nir/s) + 1 - Math.sqrt((2 * (samples.nir/s) + 1) * (2 * (samples.nir/s) + 1) - 8 * ((samples.nir/s) - (samples.red/s)))) / 2;
            let ndre = (samples.nir - samples.rededge) / (samples.nir + samples.rededge);
            let Clre = ((samples.nir / samples.rededge)-1);
            let SRre = (samples.nir / samples.rededge);
            let red = samples.red/s;
            let green = samples.green/s;
            let blue = samples.blue/s;
            let nir = samples.nir/s;
            let rededge = samples.rededge/s;
            return {
                ndvi: [ndvi],
                gndvi: [gndvi],
                evi: [evi],
                savi: [savi],
                msavi: [msavi],
                ndre: [ndre],
                Clre: [Clre],
                SRre: [SRre],
                red: [red],
                green: [green],
                blue: [blue],
                nir: [nir],
                rededge: [rededge],
                dataMask: [samples.dataMask]
            };
        }
        """

        # Create a list to hold requests
        ndvi_requests = []

        # Iterate over each row in the GeoDataFrame
        for index, row in gdf.iterrows():
            # Define the time interval for each row (1 day interval around Image_Acquisition_date)
            start_date = (row['Img_date'] - timedelta(days=6)).strftime('%Y-%m-%d')
            end_date = (row['Img_date']).strftime('%Y-%m-%d')

            time_interval = (start_date, end_date)

            # Define the aggregation settings
            aggregation = SentinelHubStatistical.aggregation(
                evalscript=evalscript, time_interval=time_interval, aggregation_interval="P1D", resolution=(resx, resy)
            )

            histogram_calculations = {
                "ndvi": {"histograms": {"default": {"nBins": 20, "lowEdge": -1.0, "highEdge": 1.0}}},
                "gndvi": {"histograms": {"default": {"nBins": 20, "lowEdge": -1.0, "highEdge": 1.0}}},
                "evi": {"histograms": {"default": {"nBins": 20, "lowEdge": -1.0, "highEdge": 1.0}}},
                "savi": {"histograms": {"default": {"nBins": 20, "lowEdge": -1.0, "highEdge": 1.0}}},
                "msavi": {"histograms": {"default": {"nBins": 20, "lowEdge": -1.0, "highEdge": 1.0}}},
                "ndre": {"histograms": {"default": {"nBins": 20, "lowEdge": -1.0, "highEdge": 1.0}}},
                "Clre": {"histograms": {"default": {"nBins": 20, "lowEdge": -1.0, "highEdge": 1.0}}},
                "SRre": {"histograms": {"default": {"nBins": 20, "lowEdge": -1.0, "highEdge": 1.0}}},
                "red": {"histograms": {"default": {"nBins": 20, "lowEdge": -1.0, "highEdge": 1.0}}},
                "green": {"histograms": {"default": {"nBins": 20, "lowEdge": -1.0, "highEdge": 1.0}}},
                "blue": {"histograms": {"default": {"nBins": 20, "lowEdge": -1.0, "highEdge": 1.0}}},
                "nir": {"histograms": {"default": {"nBins": 20, "lowEdge": -1.0, "highEdge": 1.0}}},
                "rededge": {"histograms": {"default": {"nBins": 20, "lowEdge": -1.0, "highEdge": 1.0}}},
            }

            # Define the request
            request = SentinelHubStatistical(
                aggregation=aggregation,
                input_data=[input_data],
                geometry=Geometry(row['Coordinates'], crs=CRS.WGS84),
                calculations=histogram_calculations,
                config=config,
            )
            ndvi_requests.append(request)

        status = f"{len(ndvi_requests)} Statistical API requests prepared"
        logger.info(status)        
        print(status)

        # Download the data
        # "SHRateLimitWarning: Download rate limit hit" - https://www.planet.com/pricing/?tab=platform
        # -- 600 requests per minute, 1000 processing units per minute for 'Enterprise S'
        download_requests = [ndvi_request.download_list[0] for ndvi_request in ndvi_requests]

        client = SentinelHubStatisticalDownloadClient(config=config)

        ndvi_stats = client.download(download_requests)

        status = f"{len(ndvi_stats)} Results from the Statistical API"
        logger.info(status)        
        print(status)

        # --------- Aligning VIs with heights -----------
        # **** moved stats2df() function to a separate function below
        ndvi_dfs = [
            stats2df(polygon_stats, original_index=index, Polygon_coordinates=row.Coordinates)
            for index, (polygon_stats, row) in enumerate(zip(ndvi_stats, gdf.itertuples()))
        ]

        combined_df = pd.concat(ndvi_dfs, ignore_index=True)
        #print(combined_df)

        combined1_df = combined_df
        # Convert interval_to to datetime for proper sorting
        combined1_df["interval_to"] = pd.to_datetime(combined1_df["interval_to"])
        # Assign rank within each original_index group based on interval_to descending
        combined1_df["rank"] = combined1_df.groupby("original_index")["interval_to"].rank(method="dense", ascending=False)
        # Filter only rows where rank == 1
        df_filtered = combined1_df[combined1_df["rank"] == 1]
        # Drop rank column , since it not needed going forward
        df_filtered = df_filtered.drop(columns=["rank"])

        gdf.reset_index(drop=True, inplace=True)

        #aligning VIs with existing df
        # Assuming 'original_index' is included in both dataframes
        final_df = gdf.merge(combined_df, left_index=True, right_on='original_index', how='left')

        # Drop 'original_index' if not needed anymore
        final_df = final_df.drop(columns=['original_index'])

        #choosing only required columns for the model
        input_columns = ["farmid","farm","paddockid","paddock","paddock_coordinates","timestamp","lat","lng","Coordinates",
            "interval_from","interval_to"]

        #***** set a dictionary for multiple uses
        model_columns_dict = {
            'mean_height':'MeanHeight(mm)',
            'evi_B0_mean': 'EVI_mean',
            'ndvi_B0_mean': 'NDVI_mean',
            'gndvi_B0_mean': 'GNDVI_mean',
            'savi_B0_mean': 'SAVI_mean',
            'msavi_B0_mean': 'MSAVI_mean',
            'ndre_B0_mean': 'NDRE_mean',
            'Clre_B0_mean': 'CLRE_mean',
            'SRre_B0_mean': 'SRre_mean',
            'red_B0_mean': 'red_mean',
            'green_B0_mean': 'green_mean',
            'blue_B0_mean': 'blue_mean',
            'nir_B0_mean': 'nir_mean',
            'rededge_B0_mean': 'rededge_mean'
        }
        model_columns = list(model_columns_dict.keys())
        
        model_df = final_df[input_columns + model_columns]
        
        logger.info("Start biomass model calculation")
        
        #***** store data in a json array to return
        input_columns = ["farmid", "paddockid", "lat", "lng", "interval_from","interval_to"]
        df_json = model_df[input_columns + model_columns].to_json(orient="records", default_handler=str)
        gw_model_result = {
            "ndvi": json.loads(df_json)
        }

        #***** remove missing/NaN values from df (drop 'inplace' flag to prevent Panda SettingWithCopyWarning)
        model_df = model_df.dropna(subset=model_columns)

        model_df.reset_index(drop=True, inplace=True)

        #***** rename columns for model: use a dictionary defined above
        model_df = model_df.rename(columns=model_columns_dict)
        model_df['JulianDate'] = pd.to_datetime(model_df['timestamp'], errors='coerce').dt.dayofyear

        # ----------- Predicting Biomass and stroring it for each GPS point -----------
        # This is a PastureCast-v1.0.0 model currently we are using, which is present in below cell of the script
        # passing the dataset through the model and getting predictions
        import joblib

        # Define file paths for the model and scaler
        model_path = "saved_models/PastureCast-v1.1.1.joblib"  # Update to actual path if different
        preprocessor_path = "saved_models/Preprocessor-v1.1.1.joblib"  # Ensure this matches saved version

        # Load the trained model and scaler
        loaded_model = joblib.load(model_path)
        loaded_preprocessor = joblib.load(preprocessor_path)

        features= ['MeanHeight(mm)', 'Julian_Cos',  'SAVI_mean', 'EVI_mean', 'NDVI_mean', 'NDRE_mean', 'Avg_21D_SWB_pct']
        X_new_preprocessed = loaded_preprocessor.transform(model_df[features])
        y_new_pred = loaded_model.predict(X_new_preprocessed)
        model_df["PredictedBiomass(kg/ha)"] = y_new_pred
        
        #***** get mean model output values for each paddock
        unique_paddocks = df['paddockid'].unique()
        paddock_biomass = []
        for pid in unique_paddocks:
            paddock_model_df = model_df[model_df['paddockid'] == pid]
                                        
            # averaging the predictions to get final biomass
            mean_biomass = paddock_model_df["PredictedBiomass(kg/ha)"].mean()
            mean_height = paddock_model_df["MeanHeight(mm)"].mean()
            
            paddock_biomass.append({
                "paddockid": pid,
                "biomass": float(mean_biomass),
                "mean_height": float(mean_height)
            })

        gw_model_result["paddocks"] = paddock_biomass
        logger.info('Biomass calculations complete')

        # --------- Function to define color map with 9 intervals based on optimum_min_cover & optimum_max_cover -----------
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        import numpy as np

        # Define the optimum cover limits (can be dynamic)
        optimum_min_cover = 3200
        optimum_max_cover = 5300

        # Define 9 evenly spaced values between the range
        num_intervals = 9
        interval_values = np.linspace(optimum_min_cover, optimum_max_cover, num_intervals)

        # Define corresponding RGB colors (QGIS-like green to red gradient)
        color_rgb_list = [
            (227, 230, 87),
            (158, 227, 174),
            (130, 217, 155),
            (100, 204, 135),
            (70, 189, 115),
            (65, 171, 93),
            (35, 139, 69),
            (0, 109, 44),
            (235, 64, 52)
        ]

        # Normalize RGB values to [0, 1] range
        colors = [(r/255, g/255, b/255) for r, g, b in color_rgb_list]

        # Normalize positions for colormap interpolation (0 to 1)
        positions = [(val - optimum_min_cover) / (optimum_max_cover - optimum_min_cover) for val in interval_values]

        # Create custom colormap
        qgis_cmap = LinearSegmentedColormap.from_list("qgis_colormap", list(zip(positions, colors)), N=256)

        # Create normalization object
        norm = Normalize(vmin=optimum_min_cover, vmax=optimum_max_cover)
        
        # ---------- Biomass map with Krigging for Farm  -------------
        import geopandas as gpd
        import pandas as pd
        import numpy as np
        from shapely import wkt
        from shapely.geometry import Point
        from pykrige.ok import OrdinaryKriging
        import matplotlib.pyplot as plt
        from rasterio import features
        from rasterio.transform import from_origin
        import rioxarray
        import contextily as ctx

        #using the predicted biomass of each point
        df = model_df

        # Convert df to GeoDataFrame
        gdf_points = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df['lng'], df['lat']),
            crs='EPSG:4326'
        ).to_crs(epsg=3857)

        # Get unique paddocks
        unique_paddocks = df[['timestamp','farm', 'paddock', 'paddock_coordinates']].drop_duplicates()
        unique_paddocks['geometry'] = unique_paddocks['paddock_coordinates'].apply(wkt.loads)
        paddock_gdf = gpd.GeoDataFrame(unique_paddocks, geometry='geometry', crs='EPSG:4326').to_crs(epsg=3857)

        # Determine bounds that cover all paddocks
        total_bounds = paddock_gdf.total_bounds  # (minx, miny, maxx, maxy)
        grid_res = 5  # meters
        minx, miny, maxx, maxy = total_bounds
        gridx = np.arange(minx, maxx, grid_res)
        gridy = np.arange(miny, maxy, grid_res)
        width = len(gridx)
        height = len(gridy)
        transform = from_origin(minx, maxy, grid_res, grid_res)

        # Master interpolation array
        master_interp = np.full((height, width), np.nan)

        # Loop over each paddock
        for idx, row in paddock_gdf.iterrows():
            paddock_name = row['paddock']
            paddock_poly = row['geometry']
            
            status = f"Processing paddock: {paddock_name}"
            logger.info(status)
            print(status)

            # Clip points to this paddock
            points_in_paddock = gdf_points[gdf_points.geometry.within(paddock_poly)]

            if points_in_paddock.empty or len(points_in_paddock) < 3:
                status = f" - Not enough points in paddock {paddock_name}, skipping."
                logger.info(status)
                print(status)
                continue

            # Extract coordinates and values
            x = points_in_paddock.geometry.x.values
            y = points_in_paddock.geometry.y.values
            z = points_in_paddock['PredictedBiomass(kg/ha)'].values

            # Kriging
            try:
                OK = OrdinaryKriging(x, y, z, variogram_model='linear', verbose=False, enable_plotting=False)
                z_interp, ss = OK.execute("grid", gridx, gridy)
            except Exception as e:
                status = f" - Kriging failed for {paddock_name}: {e}"
                logger.info(status)
                print(status)
                continue

            # Mask the result to this paddock
            interp_arr = np.flipud(z_interp)
            mask = features.geometry_mask(
                geometries=[paddock_poly],
                transform=transform,
                invert=True,
                out_shape=interp_arr.shape
            )
            interp_masked = np.where(mask, interp_arr, np.nan)

            # Merge into master raster (take values only where master is NaN)
            master_interp = np.where(np.isnan(master_interp), interp_masked, master_interp)
            
            
        #Storing the Krigging values into GeoTiff image
        # Plot everything
        import rasterio
        import os

        # Extract farm and date info for filename/title
        farm_name = paddock_gdf['farm'].iloc[0].replace(" ", "_")
        date_str = pd.to_datetime(paddock_gdf['timestamp'].iloc[0]).strftime("%m_%d_%Y")

        # Build dynamic output path and title
        output_dir = "biomass_maps"
        output_filename = f"farm_{farmid}_{date_str}.tif"
        output_tif_path = f"{output_dir}/{output_filename}"

        # Safely delete previous
        if os.path.exists(output_tif_path):
            os.remove(output_tif_path)

        # Save new file
        with rasterio.open(
            output_tif_path, 'w',
            driver='GTiff',
            height=master_interp.shape[0],
            width=master_interp.shape[1],
            count=1,
            dtype=master_interp.dtype,
            crs='EPSG:3857',
            transform=transform,
            nodata=np.nan
        ) as dst:
            dst.write(master_interp, 1)

        status = f"New GeoTIFF written: {output_tif_path}"
        logger.info(status)
        print(status)

        #Using GeoTiff image to overlay on ESRI world map
        rds = rioxarray.open_rasterio(output_tif_path, masked=True)

        # Show raster
        # rds.plot(ax=ax, cmap="YlGn", vmin=2800, vmax=5600, add_colorbar=True)
        fig, ax = plt.subplots(figsize=(12, 12))

        # Plot biomass raster
        rds.plot(ax=ax, cmap=qgis_cmap, norm=norm, add_colorbar=True)

        # Plot basemap
        ctx.add_basemap(ax, crs=rds.rio.crs, source=ctx.providers.Esri.WorldImagery)

        # Plot paddock boundaries
        paddock_gdf.boundary.plot(ax=ax, edgecolor='white', linewidth=1)

        # Add paddock name labels at centroids
        for idx, row in paddock_gdf.iterrows():
            centroid = row['geometry'].centroid
            ax.text(
                centroid.x, centroid.y,
                row['paddock'],  
                fontsize=9,
                color='white',
                ha='center',
                va='center',
                weight='bold',
                bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2')
            )

        # Final layout
        ax.set_title(f"Interpolated Biomass for {farm_name.replace('_',' ')} farm on {date_str}")
        ax.axis('off')
        plt.tight_layout()
        #plt.show()

        # Save the plot as a JPEG file
        output_jpg_path = output_tif_path.replace(".tif", ".jpg")
        plt.savefig(output_jpg_path, dpi=300)     

        status = f"Map saved to JPG format: {output_jpg_path}"
        logger.info(status)
        print(status)
        
        # Add to result
        gw_model_result["biomass_map"] = output_jpg_path

        # ---------- All done -------------
        time_end = datetime.now()
        durationTime = time_end - time_start
        duration = str(durationTime)

        logger.info(f'End gw_model_v2("{farmid}", "{measure_date}"), duration={duration}')
        
        # write json output to a file
        json_file = get_jsonfile(farmid, measure_date)
        with open(json_file, "w") as jsonfile:
            json.dump(gw_model_result, jsonfile, indent=4)
        
        logger.info(f'Output json file: {json_file}')
        return json.dumps(gw_model_result)
    
    except Exception as e:
        logger.exception(str(e))
        return None
    
    finally:
        pass

# Return JSON data from a file
def gw_model_json( farmid, measure_date ):
    import json
    from gw_logger import get_gw_logger
    
    logger = get_gw_logger()
    
    try:
        json_file = get_jsonfile(farmid, measure_date)
        with open(json_file, "r") as jsonfile:
            data = json.load(jsonfile)
        
        return json.dumps(data)
    
    except Exception as e:
        logger.exception(str(e))
        return None
    
    finally:
        pass

# Optional: process the results as needed
#--------------
#converting statistical output from above script into df with polygon coordinates
def stats2df(stats_data, original_index, Polygon_coordinates):
    import pandas as pd
    from sentinelhub import (
        parse_time
    )
    
    """Transform Statistical API response into a pandas.DataFrame"""
    df_data = []

    for single_data in stats_data["data"]:
        df_entry = {}
        is_valid_entry = True
        df_entry["interval_from"] = parse_time(single_data["interval"]["from"]).date()
        df_entry["interval_to"] = parse_time(single_data["interval"]["to"]).date()

        # Add the subplot coordinates and original index to the entry
        df_entry["original_index"] = original_index
        df_entry["Polygon_coordinates"] = Polygon_coordinates

        for output_name, output_data in single_data["outputs"].items():
            for band_name, band_values in output_data["bands"].items():
                band_stats = band_values["stats"]
                if band_stats["sampleCount"] == band_stats["noDataCount"]:
                    is_valid_entry = False
                    break

                for stat_name, value in band_stats.items():
                    col_name = f"{output_name}_{band_name}_{stat_name}"
                    if stat_name == "percentiles":
                        for perc, perc_val in value.items():
                            perc_col_name = f"{col_name}_{perc}"
                            df_entry[perc_col_name] = perc_val
                    else:
                        df_entry[col_name] = value

        if is_valid_entry:
            df_data.append(df_entry)

    return pd.DataFrame(df_data)


def get_jsonfile(farmid, measure_date):
    return r"..\logs\api-python\farm-" + farmid + "-" + measure_date.replace("/", "-") + ".json"


# Execute the function
#gw_model_v2('913', '05/26/2025')
#gw_model_json('913', '05/26/2025')
