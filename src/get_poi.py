"""
Get POIs from google map
"""
import configparser
import json  # pylint: disable=W0611
import time

import geopandas as gpd
import googlemaps
import matplotlib.pyplot as plt  # pylint: disable=W0611
import numpy as np
import pandas as pd
from shapely import geometry


def grid_cluster_recursive(grid):
    """
    Recursively obtain the sub-grid if the number of points exceeds 60.
    """
    geom_array = []

    # 把方格四格化
    geom_array = np.array([])
    centroid_x = grid.geometry.centroid.x
    centroid_y = grid.geometry.centroid.y
    min_x, min_y, max_x, max_y = grid.geometry.bounds

    # upper left
    geom_ul = geometry.Polygon([(min_x,centroid_y), (min_x, max_y),
                                (centroid_x, max_y), (centroid_x, centroid_y), (min_x, centroid_y)])
    # upper right
    geom_ur = geometry.Polygon([(centroid_x,centroid_y), (centroid_x, max_y),
                                (max_x, max_y), (max_x, centroid_y), (centroid_x, centroid_y)])
    # bottom left
    geom_bl = geometry.Polygon([(min_x,min_y), (min_x, centroid_y),
                                (centroid_x, centroid_y), (centroid_x, min_y), (min_x, min_y)])
    # bottom right
    geom_br = geometry.Polygon([(centroid_x,min_y), (centroid_x, centroid_y),
                                (max_x, centroid_y), (max_x, min_y), (centroid_x, min_y)])

    geom_array = np.append(geom_array,[geom_ul,geom_ur,geom_bl,geom_br])

    sub_grid = gpd.GeoDataFrame(geometry=geom_array).set_crs('EPSG:3826')
    sub_grid['length'] = np.sqrt(sub_grid.geometry.area)
    sub_grid['radius'] = (sub_grid['length']/2) * np.sqrt(2)
    sub_grid['radius'] = sub_grid['radius'].apply(lambda x:round(x,3))
    sub_grid['centroid'] = sub_grid.geometry.centroid
    sub_grid['centroid_84'] = sub_grid.centroid.to_crs('EPSG:4326')

    # geom_centroid_array = np.append(geom_centroid_array,
    #                           [geom_ul_centroid,geom_ur_centroid,
    #                                geom_bl_centroid,geom_br_centroid])
    # geom_radious_array = np.append(geom_radious_array,
    #                           [geom_ul_radious,geom_ur_radious,
    #                                geom_bl_radious,geom_br_radious])
    # get_centroid
    # get_radius
    return sub_grid

results = []  # nearby list
grid_num = 1  # pylint: disable=C0103
def search_poi(row, poi_type, keyword, gmaps): # pylint:  disable=W0621
    global grid_num
    # b = 1
    # for obj in point_list:
    # point_reproject = point_list.to_crs('EPSG:4326')
    loc = {'lat': row.centroid_84.y,'lng': row.centroid_84.x}
    query_result = gmaps.places_nearby(keyword=keyword, location=loc,
                                       radius=row.radius, type=poi_type)
    results.extend(query_result['results'])
    print(f"找到半徑{row.radius}公尺內{keyword} {str(len(query_result['results']))}間")

    a = 1
    while query_result.get('next_page_token') and row.radius>=0.1:
        time.sleep(2)
        query_result = gmaps.places_nearby(page_token=query_result['next_page_token'])
        results.extend(query_result['results'])
        print(f"翻{a}頁，找到半徑{row.radius}公尺內{keyword} {str(len(query_result['results']))}間")
        if (a == 2) and (len(query_result['results']) == 20):
            # Split the grid and redo the search process for each sub-grid
            print('reach max limit 60 results!, splitting the grid into four sub-parts')
            sub_grids = grid_cluster_recursive(row)
            for index, sub_grid in sub_grids.iterrows():
                print(f'searching the {index+1}/4 sub-grid')
                search_poi(sub_grid, poi_type, keyword, gmaps)

        a += 1

    grid_num += 1
    print(f'進度: {grid_num}/{grid_final.shape[0]}')
    print('========================================')
    # b += 1
    # return results

def elt(results): # pylint: disable=W0621
    # ETL
    temps = []
    for place in results:
        # pylint: disable=W0702
        try:
            temp = {'id': place['place_id'],
                    'keyword': '',
                    'Name': place['name'],
                    'lng': place['geometry']['location']['lng'],
                    'lat': place['geometry']['location']['lat'],
                    'addr': place['vicinity'],
                    'rating_num': place.get('user_ratings_total'),
                    'rating': place.get('rating'),
                    'price level': place.get('price_level')}
        except:
            print('no vicinty')
            temp = {'id': place['place_id'],
                    'keyword': '',
                    'Name': place['name'],
                    'lng': place['geometry']['location']['lng'],
                    'lat': place['geometry']['location']['lat'],
                    'rating_num': place.get('user_ratings_total'),
                    'rating': place.get('rating'),
                    'price level': place.get('price_level')}
        temps.append(temp)
    poi = pd.DataFrame(temps)
    poi = poi.drop_duplicates('id')  # 去重覆
    poi = poi.drop_duplicates('addr')
    return poi



if __name__ == '__main__':
    # Import data
    grid_df = gpd.read_file(r'../data/raw/FET_2023_grid_97.geojson')
    grid_final = gpd.read_file(r'../data/raw/grid_6000_cluster_1000.shp')
    grid_final['length'] = np.sqrt(grid_final.geometry.area)
    grid_final['radius'] = ((grid_final['length']/2) * np.sqrt(2)).round(3)
    grid_final['centroid'] = grid_final.geometry.centroid
    grid_final['centroid_84'] = grid_final.centroid.to_crs('EPSG:4326')

    # Read API
    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')
    API = config.get('API','API')

    gmaps = googlemaps.Client(key=API)

    POI_TYPE = 'store'
    KEYWORD = ''
    grid_final.apply(search_poi,axis=1,args=(POI_TYPE, KEYWORD, gmaps))
