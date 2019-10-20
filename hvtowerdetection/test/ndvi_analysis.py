from functools import partial
import numpy as np
import fiona
import pyproj
from osgeo import gdal as gd
import geopandas as gpd
from shapely.ops import transform
from shapely.geometry import Point, Polygon


## add fiona support
fiona.drvsupport.supported_drivers['kml'] = 'rw' # enable KML support which is disabled by default
fiona.drvsupport.supported_drivers['KML'] = 'rw' # enable KML support which is disabled by default
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'


# define projection
proj_wgs84 = pyproj.Proj(init='epsg:4326')


class NdviAnalysis:
    def __init__(self, ms_path, object_shp_path, radius, size_x, size_y):
        self.ms_path = ms_path
        self.object_shp_path = object_shp_path
        self.radius = radius
        self.size_x = size_x
        self.size_y = size_y

    # read ndvi tiff file
    def read_ms(self, path):
        ds_ms = gd.Open(path)
        return ds_ms

    # convert latlong coordinates as pixels
    def convert_latlong_as_pixel(self, ds_ms, point):
        gt = ds_ms.GetGeoTransform()
        row = int((point.x - gt[0])/gt[1])
        col = int((point.y - gt[3])/gt[5])
        return row, col

    # create a circle around a lat long
    def geodesic_point_buffer(self, lat, lon):
        # Azimuthal equidistant projection
        aeqd_proj = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
        project = partial(
            pyproj.transform,
            pyproj.Proj(aeqd_proj.format(lat=lat, lon=lon)),
            proj_wgs84)
        buf = Point(0, 0).buffer(self.radius)  # radius in metres
        return transform(project, buf).exterior.coords[:]

    # return average ndvi based on xy offset
    def return_avg_ndvi(self, ds_ms, xoff, yoff):
        red = ds_ms.GetRasterBand(5).ReadAsArray(xoff, yoff, self.size_x, self.size_y).astype('int16')
        nir1 = ds_ms.GetRasterBand(7).ReadAsArray(xoff, yoff, self.size_x, self.size_y).astype('int16')
        # calculate ndvi
        ndvi_array = (nir1 - red)/(nir1 + red)
        ndvi_array[~ np.isfinite(ndvi_array)] = 0
        return np.average(ndvi_array)

    def encroachment_analysis(self, save_path):
        ds_ms = self.read_ms(self.ms_path)
        # read predicted hv towers kml/shp
        df_hv = gpd.read_file(self.object_shp_path)
        # print (len(df_hv), df_hv.crs)
        df_hv = df_hv[['id', 'geometry']]
        df_hv['centroid'] = df_hv.centroid

        # loop through all towers and create a circle of radius and convert as shapely polygon
        geom_circle = []
        for i in range(len(df_hv)):
            pt = df_hv['centroid'].iloc[i]
            lat = pt.y
            lon = pt.x
            polycircle = Polygon(self.geodesic_point_buffer(lat, lon))
            geom_circle.append(polycircle)

        # update geometry column
        df_hv['geometry'] = geom_circle

        # loop through all towers and calculate avaerage ndvi
        avg_ndvi = []
        for j in range(len(df_hv)):
            row, col = self.convert_latlong_as_pixel(ds_ms, df_hv['centroid'].iloc[j])
            xoff = row
            yoff = col
            print('\nrow col: {} {}'.format(row, col))
            try:
                avg_ndvi.append(self.return_avg_ndvi(ds_ms, xoff, yoff))
            except:
                avg_ndvi.append(0)
                continue

        # add average ndvi column in dataframe
        df_hv['avg_ndvi'] = avg_ndvi

        # save as shapefile
        df_hv = df_hv[['id', 'avg_ndvi', 'geometry']]
        df_hv.to_file(os.path.join(save_path, 'hvtower_ndvi_encroachment.shp'), driver='ESRI Shapefile')
        df_hv.to_file(os.path.join(save_path, 'hvtower_ndvi_encroachment.kml'), driver='KML')
