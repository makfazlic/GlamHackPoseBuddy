## Glam Hack main file

#Loading data, checking against input

import pose_extract as pe
import matplotlib as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import json
import cv2
import os
import pymongo

CLIENT = pymongo.MongoClient("mongodb+srv://admin:QJ1KYlgo3kP8AYCk@cluster0.b6tbfym.mongodb.net/test")
DB = CLIENT["poseBuddy"]
COL = DB["metadata"]

def get_all_from_db(col):
    items = col.find()
    return [item for item in items]

def get_query(query, col):
    return 

# Get images and filenames from folder
def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

def feature_error(vector_a, vector_b):
    length = 0
    length += (vector_a["x"] - vector_b["x"])*(vector_a["x"] - vector_b["x"])
    length += (vector_a["y"] - vector_b["y"])*(vector_a["y"] - vector_b["y"])
    length += (vector_a["z"] - vector_b["z"])*(vector_a["z"] - vector_b["z"])
    vector_visibility = (vector_a["visibility"] + vector_b["visibility"])/2
    if vector_visibility > 0.5:
        return np.sqrt(length) * vector_visibility, vector_visibility
    else:
        return 0, 0

def compare_images(our_points, our_vectors, museum_painting):
    painting_points = museum_painting["Joints"]
    painting_vectors = museum_painting["Vectors"]
    this_image_error = 0
    seen_vectors = 0
    for i in range(13):
        i_str = str(i)
        fe, seen = feature_error(our_vectors[i_str], painting_vectors[i_str])
        if ((seen != 0)):
            seen_vectors += seen
        this_image_error += fe
    this_image_error = this_image_error
    return this_image_error

def read_and_update_query(vectors):

    print(vectors[7])

    resp = CLIENT["poseBuddy"]["metadata_with_array"].aggregate(
    [
    {
        '$project': {
            'Filename': 1,
            'shoulders': {
                '$arrayElemAt': [
                    '$Vectors', 0
                ]
            }, 
            'rightUpperArm': {
                '$arrayElemAt': [
                    '$Vectors', 1
                ]
            }, 
            'leftUpperArm': {
                '$arrayElemAt': [
                    '$Vectors', 2
                ]
            }, 
            'rightForearm': {
                '$arrayElemAt': [
                    '$Vectors', 3
                ]
            }, 
            'leftForearm': {
                '$arrayElemAt': [
                    '$Vectors', 4
                ]
            }, 
            'hip': {
                '$arrayElemAt': [
                    '$Vectors', 5
                ]
            }, 
            'rightWing': {
                '$arrayElemAt': [
                    '$Vectors', 6
                ]   
            },
            'rightUpperLeg': {
                '$arrayElemAt': [
                    '$Vectors', 8
                ]
            }, 
            'leftUpperLeg': {
                '$arrayElemAt': [
                    '$Vectors', 9
                ]
            }, 
            'rightShin': {
                '$arrayElemAt': [
                    '$Vectors', 10
                ]
            }, 
            'leftShin': {
                '$arrayElemAt': [
                    '$Vectors', 11
                ]
            }
        }
    }, 
 
    {
        '$project': {
            'Filename': 1,
            'errorShoulders': {
                '$sqrt': {
                    '$add': [
                        {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$shoulders.x', vectors[0]['x']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$shoulders.x', vectors[0]['x']
                                    ]
                                }
                            ]
                        }, {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$shoulders.y', vectors[0]['y']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$shoulders.y', vectors[0]['y']
                                    ]
                                }
                            ]
                        }, {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$shoulders.z', vectors[0]['z']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$shoulders.z', vectors[0]['z']
                                    ]
                                }
                            ]
                        }
                    ]
                }
            }, 
            'errorRightUpperArm': {
                '$sqrt': {
                    '$add': [
                        {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$rightUpperArm.x', vectors[1]['x']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$rightUpperArm.x', vectors[1]['x']
                                    ]
                                }
                            ]
                        }, {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$rightUpperArm.y', vectors[1]['y']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$rightUpperArm.y', vectors[1]['y']
                                    ]
                                }
                            ]
                        }, {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$rightUpperArm.z', vectors[1]['z']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$rightUpperArm.z', vectors[1]['z']
                                    ]
                                }
                            ]
                        }
                    ]
                }
            }, 
            'errorLeftUpperArm': {
                '$sqrt': {
                    '$add': [
                        {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$leftUpperArm.x', vectors[2]['x']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$leftUpperArm.x', vectors[2]['x']
                                    ]
                                }
                            ]
                        }, {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$leftUpperArm.y', vectors[2]['y']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$leftUpperArm.y', vectors[2]['y']
                                    ]
                                }
                            ]
                        }, {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$leftUpperArm.z', vectors[2]['z']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$leftUpperArm.z', vectors[2]['z']
                                    ]
                                }
                            ]
                        }
                    ]
                }
            }, 
            'errorRightForearm': {
                '$sqrt': {
                    '$add': [
                        {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$rightForearm.x', vectors[3]['x']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$rightForearm.x', vectors[3]['x']
                                    ]
                                }
                            ]
                        }, {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$rightForearm.y', vectors[3]['y']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$rightForearm.y', vectors[3]['y']
                                    ]
                                }
                            ]
                        }, {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$rightForearm.z', vectors[3]['z']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$rightForearm.z', vectors[3]['z']
                                    ]
                                }
                            ]
                        }
                    ]
                }
            }, 
            'errorLeftForearm': {
                '$sqrt': {
                    '$add': [
                        {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$leftForearm.x', vectors[4]['x']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$leftForearm.x', vectors[4]['x']
                                    ]
                                }
                            ]
                        }, {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$leftForearm.y', vectors[4]['y']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$leftForearm.y', vectors[4]['y']
                                    ]
                                }
                            ]
                        }, {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$leftForearm.z', vectors[4]['z']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$leftForearm.z', vectors[4]['z']
                                    ]
                                }
                            ]
                        }
                    ]
                }
            }, 
            'errorHip': {
                '$sqrt': {
                    '$add': [
                        {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$hip.x', vectors[5]['x']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$hip.x', vectors[5]['x']
                                    ]
                                }
                            ]
                        }, {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$hip.y', vectors[5]['y']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$hip.y', vectors[5]['y']
                                    ]
                                }
                            ]
                        }, {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$hip.z', vectors[5]['z']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$hip.z', vectors[5]['z']
                                    ]
                                }
                            ]
                        }
                    ]
                }
            }, 
            'errorRightWing': {
                '$sqrt': {
                    '$add': [
                        {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$rightWing.x', vectors[6]['x']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$rightWing.x', vectors[6]['x']
                                    ]
                                }
                            ]
                        }, {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$rightWing.y', vectors[6]['y']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$rightWing.y', vectors[6]['y']
                                    ]
                                }
                            ]
                        }, {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$rightWing.z', vectors[6]['z']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$rightWing.z', vectors[6]['z']
                                    ]
                                }
                            ]
                        }
                    ]
                }
            }, 
            'errorLeftWing': {
                '$sqrt': {
                    '$add': [
                        {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$leftWing.x', vectors[7]['x']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$leftWing.x', vectors[7]['x']
                                    ]
                                }
                            ]
                        }, {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$leftWing.y', vectors[7]['y']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$leftWing.y', vectors[7]['y']
                                    ]
                                }
                            ]
                        }, {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$leftWing.z', vectors[7]['z']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$leftWing.z', vectors[7]['z']
                                    ]
                                }
                            ]
                        }
                    ]
                }
            }, 
            'errorRightUpperLeg': {
                '$sqrt': {
                    '$add': [
                        {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$rightUpperLeg.x', vectors[8]['x']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$rightUpperLeg.x', vectors[8]['x']
                                    ]
                                }
                            ]
                        }, {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$rightUpperLeg.y', vectors[8]['y']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$rightUpperLeg.y', vectors[8]['y']
                                    ]
                                }
                            ]
                        }, {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$rightUpperLeg.z', vectors[8]['z']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$rightUpperLeg.z', vectors[8]['z']
                                    ]
                                }
                            ]
                        }
                    ]
                }
            }, 
            'errorLeftUpperLeg': {
                '$sqrt': {
                    '$add': [
                        {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$leftUpperLeg.x', vectors[9]['x']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$leftUpperLeg.x', vectors[9]['x']
                                    ]
                                }
                            ]
                        }, {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$leftUpperLeg.y', vectors[9]['y']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$leftUpperLeg.y', vectors[9]['y']
                                    ]
                                }
                            ]
                        }, {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$leftUpperLeg.z', vectors[9]['z']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$leftUpperLeg.z', vectors[9]['z']
                                    ]
                                }
                            ]
                        }
                    ]
                }
            }, 
            'errorRightShin': {
                '$sqrt': {
                    '$add': [
                        {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$rightShin.x', vectors[10]['x']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$rightShin.x', vectors[10]['x']
                                    ]
                                }
                            ]
                        }, {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$rightShin.y', vectors[10]['y']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$rightShin.y', vectors[10]['y']
                                    ]
                                }
                            ]
                        }, {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$rightShin.z', vectors[10]['z']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$rightShin.z', vectors[10]['z']
                                    ]
                                }
                            ]
                        }
                    ]
                }
            }, 
            'errorLeftShin': {
                '$sqrt': {
                    '$add': [
                        {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$leftShin.x', vectors[11]['x']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$leftShin.x', vectors[11]['x']
                                    ]
                                }
                            ]
                        }, {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$leftShin.y', vectors[11]['y']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$leftShin.y', vectors[11]['y']
                                    ]
                                }
                            ]
                        }, {
                            '$multiply': [
                                {
                                    '$subtract': [
                                        '$leftShin.z', vectors[11]['z']
                                    ]
                                }, {
                                    '$subtract': [
                                        '$leftShin.z', vectors[11]['z']
                                    ]
                                }
                            ]
                        }
                    ]
                }
            }
        }
    }, 

    {
        '$addFields': {
            'errorLeftWing': 0
        }
    },
    {
        '$project': {
            'Filename': 1,
            'featureError': {
                '$add': [
                    '$errorShoulders', '$errorRightUpperArm', '$errorLeftUpperArm', '$errorRightForearm', '$errorLeftForearm', '$errorHip', '$errorRightWing', '$errorLeftWing', '$errorRightUpperLeg', '$errorLeftUpperLeg', '$errorRightShin', '$errorLeftShin'
                ]
            }
        }
    }, 
    {
        '$addFields': {
            'error': {
                '$multiply': [
                    '$featureError', 1000
                ]
            }
        }
    }, {
        '$addFields': {
            'error': {
                '$convert': {
                    'input': '$error', 
                    'to': 'int'
                }
            }
        }
    }, {
        '$sort': {
            'error': 1
        }
    },
    {
        '$limit': 10
    }
]
    )

    return resp

def run():
    paintings, filenames = load_images_from_folder("./processed_images/")
    vid = cv2.VideoCapture(0)
    image = None
    while(True):
        ret, frame = vid.read()
        cv2.imshow('Take a photo by pressing P on the keyboard', pe.paint_points(frame))
      
        if cv2.waitKey(10) & 0xFF == ord('q'):
            exit()

        if cv2.waitKey(10) & 0xFF == ord('p'):
            image = frame
            break
    vid.release()
    cv2.destroyAllWindows()
    time.sleep(1)

    image_with_landmarks, my_points = pe.extract(image)
    my_vectors = pe.generate_vectors(my_points)
    their_entries = get_all_from_db(COL)
    
    compare_scores = []
    for museum_painting in their_entries:
        compare_scores.append(compare_images(my_points, my_vectors, museum_painting))
  
    
    top_5 = np.array(compare_scores).argsort()[:5]
    cv2.imshow('Your input', image_with_landmarks)
    for top in top_5:
        cv2.imshow("Your pose buddy", pe.paint_points(paintings[top]))
        if cv2.waitKey(0) & 0xFF == ord('n'):
            pass

    if cv2.waitKey(0) & 0xFF == ord('q'):
       cv2.destroyAllWindows()
       return 0     

def run_with_query():
    vid = cv2.VideoCapture(0)
    image = None
    while(True):
        ret, frame = vid.read()
        cv2.imshow('Take a photo by pressing P on the keyboard', pe.paint_points(frame))
      
        if cv2.waitKey(10) & 0xFF == ord('q'):
            exit()

        if cv2.waitKey(10) & 0xFF == ord('p'):
            image = frame
            break

    vid.release()
    cv2.destroyAllWindows()
    time.sleep(1)
    image_with_landmarks, my_points = pe.extract(image)
    cv2.imshow('This is your input', image_with_landmarks)

    my_vectors = pe.generate_vectors(my_points)
    back_vectors = read_and_update_query(my_vectors)
    for v in back_vectors:
        a = './processed_images/' + v["Filename"]
        print(v)
        img = cv2.imread(a)
        cv2.imshow('image',pe.paint_points(img))
        cv2.waitKey(0)
        break
run_with_query()