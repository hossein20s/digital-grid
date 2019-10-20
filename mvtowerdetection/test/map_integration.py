from math import sin, cos, asin, sqrt, degrees, radians
import math
import pyproj
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, shape, mapping, Point
from shapely.ops import transform
import fiona

fiona.drvsupport.supported_drivers['kml'] = 'rw'  # enable KML support which is disabled by default
fiona.drvsupport.supported_drivers['KML'] = 'rw'  # enable KML support which is disabled by default
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'


class MapIntegration:
    def __init__(self, track_ground_truth_file, object_attr_file, save_path):
        self.track_ground_truth_file = track_ground_truth_file
        self.object_attr_file = object_attr_file
        self.save_path = save_path

    # change projection for the road to EPSG:6437 (florida)
    def change_crs_road(self, road):
        project = pyproj.Transformer.from_proj(
        pyproj.Proj(init='epsg:4326'),  # source coordinate system
        pyproj.Proj(init='epsg:6437'))  # destination coordinate system (florida)

        road = transform(project.transform, road)   # apply projection
        return road

    # return road depending upon image id
    def return_road(self, df_original, image_id):
        index = df_original[df_original['fileName'] == image_id+'.jpg'].index.values[0]
        df_original = df_original[index:]
        coord = []
        for i in range(len(df_original)):
            coord.append((df_original['lng'].iloc[i], df_original['lat'].iloc[i]))
        road = LineString(coord)

        return self.change_crs_road(road)

    # calculate interpolation for the towers
    def return_poles_locations(self, df_imgid, road):
        pole_lat_long = []
        for i in range(len(df_imgid)):
            distance = float(df_imgid['depth'].iloc[i])*0.001 ## distance in km
            point = road.interpolate(distance)
            # change crs
            crs = {'init': 'epsg:6437'}
            dn = gpd.GeoDataFrame(crs=crs, geometry=[point])
            dn_new = dn.to_crs({'init': 'epsg:4326'})
            pole_lat_long.append((dn_new['geometry'].iloc[0]))

        return pole_lat_long

    # calculate general bearing
    def calculate_initial_compass_bearing(self, pointA, pointB):
        if (type(pointA) != tuple) or (type(pointB) != tuple):
            raise TypeError("Only tuples are supported as arguments")

        lat1 = math.radians(pointA[0])
        lat2 = math.radians(pointB[0])

        diffLong = math.radians(pointB[1] - pointA[1])

        x = math.sin(diffLong) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
                * math.cos(lat2) * math.cos(diffLong))

        initial_bearing = math.atan2(x, y)
        initial_bearing = math.degrees(initial_bearing)
        compass_bearing = (initial_bearing + 360) % 360

        return compass_bearing

    # calculate new lat long based on change in bearing due to poles being on left or right
    def return_new_longlat(self, img_lat_deg, img_lon_deg, initial_bearing_deg, depth, loc='right'):
        R = 6378.1  # Radius of the Earth in km
        distance_from_centerofroad = 4  # in meters
        angle = math.atan(distance_from_centerofroad/depth)
        if loc=='right':
            new_brng = radians(initial_bearing_deg) + angle  # Bearing is converted to radians.
        if loc=='left':
            new_brng = radians(initial_bearing_deg) - angle  # Bearing is converted to radians.

        d = sqrt(depth * depth + distance_from_centerofroad * distance_from_centerofroad) * 0.001  # Distance in km

        lat1 = math.radians(img_lat_deg)  # Current lat point converted to radians
        lon1 = math.radians(img_lon_deg)  # Current long point converted to radians

        lat2 = math.asin(math.sin(lat1) * math.cos(d / R) +
             math.cos(lat1) * math.sin(d / R) * math.cos(brng))

        lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(d / R) * math.cos(lat1),
                     math.cos(d / R) - math.sin(lat1) * math.sin(lat2))

        lat2 = math.degrees(lat2)
        lon2 = math.degrees(lon2)

        return (lon2, lat2)

    # main function
    def get_object_geocoordinates(self, save=True):
        # get all unique imageids on which we performed prediction
        df_poles = pd.read_csv(self.object_attr_file)
        df_original = pd.read_json(self.track_ground_truth_file)
        
        imgids = df_poles['image_id'].unique()
        final_pole_latlong = []

        counter = 1
        for imgid in imgids:
            print('counter: {}'.format(counter))
            road = self.return_road(df_original, imgid)
            df_imgid = df_poles[df_poles['image_id'] == imgid + '.jpg']

            img_lat_deg = float(df_original[df_original['fileName'] == imgid + '.jpg']['lat'].values)
            img_lon_deg = float(df_original[df_original['fileName'] == imgid + '.jpg']['lng'].values)
            pointA = (img_lat_deg, img_lon_deg)

            poles_locations = self.return_poles_locations(df_imgid, road)

            # loop on poles detected on road to incorporate direction effects (left or right)
            for p in range(len(poles_locations)):
                initial_bearing_deg = self.calculate_initial_compass_bearing(pointA, poles_locations[p])
                depth = df_imgid['depth'].iloc[p]
                loc = df_imgid['direction'].iloc[p]
                final_pole_point = self.return_new_longlat(img_lat_deg, img_lon_deg, initial_bearing_deg, depth, loc=loc)
                final_pole_latlong.append(Point(final_pole_point))
            counter += 1

        final_crs = {'init': 'epsg:4326'}
        df_final_pole = gpd.GeoDataFrame(crs=final_crs, columns=[imgids], geometry = [final_pole_latlong])

        if save:
            df_final_pole.to_file(os.path.join(self.save_path, 'distribution_poles.shp'), driver='ESRI Shapefile')
            df_final_pole.to_file(os.path.join(self.save_path, 'distribution_poles.kml'), driver='KML')


## read json file containing all details specific to an image id
# df_original = pd.read_json('/home/ubuntu/osc-tools/477468.json')
# df_poles = pd.read_csv('/home/ubuntu/poles_prediction_loc.csv')
# pathtosave = '/home/ubuntu/'
# filename = 'finalfloridapoleslocation.kml'

## call main function
# main_func(df_poles, df_original, pathtosave, filename)
