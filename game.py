import pygame
import sys
import random
import time
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import threading
import pickle
import queue

###################CAC BIEN DE XU LY ANH################
frameWidth = 640  # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75  # PROBABILITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# IMPORT THE TRAINED MODEL
#model = load_model('traffic_sign_model.h5')
pickle_in = open("new_model/model_data_1.p", "rb")
model = pickle.load(pickle_in)

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def getClassName(classNo):
    if   classNo == 0: return 'Speed Limit 20 km/h'
    elif classNo == 1: return 'Speed Limit 30 km/h'
    elif classNo == 2: return 'Speed Limit 50 km/h'
    elif classNo == 3: return 'Speed Limit 60 km/h'
    elif classNo == 4: return 'Speed Limit 70 km/h'
    elif classNo == 5: return 'Speed Limit 80 km/h'
    elif classNo == 6: return 'End of Speed Limit 80 km/h'
    elif classNo == 7: return 'Speed Limit 100 km/h'
    elif classNo == 8: return 'Speed Limit 120 km/h'
    elif classNo == 9: return 'No passing'
    elif classNo == 10: return 'No passing for vehicles over 3.5 metric tons'
    elif classNo == 11: return 'Right-of-way at the next intersection'
    elif classNo == 12: return 'Priority road'
    elif classNo == 13: return 'Yield'
    elif classNo == 14: return 'Stop'
    elif classNo == 15: return 'No vehicles'
    elif classNo == 16: return 'Vehicles over 3.5 metric tons prohibited'
    elif classNo == 17: return 'No entry'
    elif classNo == 18: return 'General caution'
    elif classNo == 19: return 'Dangerous curve to the left'
    elif classNo == 20: return 'Dangerous curve to the right'
    elif classNo == 21: return 'Double curve'
    elif classNo == 22: return 'Bumpy road'
    elif classNo == 23: return 'Slippery road'
    elif classNo == 24: return 'Road narrows on the right'
    elif classNo == 25: return 'Road work'
    elif classNo == 26: return 'Traffic signals'
    elif classNo == 27: return 'Pedestrians'
    elif classNo == 28: return 'Children crossing'
    elif classNo == 29: return 'Bicycles crossing'
    elif classNo == 30: return 'Beware of ice/snow'
    elif classNo == 31: return 'Wild animals crossing'
    elif classNo == 32: return 'End of all speed and passing limits'
    elif classNo == 33: return 'Turn right ahead'
    elif classNo == 34: return 'Turn left ahead'
    elif classNo == 35: return 'Ahead only'
    elif classNo == 36: return 'Go straight or right'
    elif classNo == 37: return 'Go straight or left'
    elif classNo == 38: return 'Keep right'
    elif classNo == 39: return 'Keep left'
    elif classNo == 40: return 'Roundabout mandatory'
    elif classNo == 41: return 'End of no passing'
    elif classNo == 42: return 'End of no passing by vehicles over 3.5 metric tons'


###################CAC BIEN CUA GAME###################
# Initialize Pygame
pygame.init()

sign_images = {
    0: 'Traffic Sign Image/20.jpg',
    1: 'Traffic Sign Image/30.jpg',
    2: 'Traffic Sign Image/50.jpg',
    3: 'Traffic Sign Image/60.jpg',
    4: 'Traffic Sign Image/70.jpg',
    5: 'Traffic Sign Image/80.jpg',
    7: 'Traffic Sign Image/100.jpg',
    8: 'Traffic Sign Image/120.jpg',
    9: 'Traffic Sign Image/nopassing.jpg',
    11: 'Traffic Sign Image/intersection.jpg',
    12: 'Traffic Sign Image/priority.jpg',
    13: 'Traffic Sign Image/yield.jpg',
    14: 'Traffic Sign Image/stop.jpg',
    15: 'Traffic Sign Image/novehicle.jpg',
    17: 'Traffic Sign Image/noentry.jpg',
    18: 'Traffic Sign Image/generalcaution.jpg',
    19: 'Traffic Sign Image/danleft.jpg',
    20: 'Traffic Sign Image/danleft.jpg',
    22: 'Traffic Sign Image/bumproad.jpg',
    23: 'Traffic Sign Image/slipperyroad.jpg',
    24: 'Traffic Sign Image/narrowright.jpg',
    25: 'Traffic Sign Image/roadwork.jpg',
    27: 'Traffic Sign Image/pedestrian.jpg',
    28: 'Traffic Sign Image/childcrossing.jpg',
    29: 'Traffic Sign Image/bicyclecrossing.jpg',
    35: 'Traffic Sign Image/aheadonly.jpg',
    40: 'Traffic Sign Image/roundabout.jpg'
}

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Colors
WHITE = (255, 255, 255)
GRAY = (169, 169, 169)
GREEN_YELLOW = (210, 170, 109)
BLUE = (135, 206, 235)
LIGHT_BLUE = (173, 216, 230)
SIDEWALK_COLOR = (210, 170, 109)  # Gray for sidewalks
TEXT_COLOR = (0, 0, 0)  # Black for text

# Create the screen object
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Infinite Driving Car Game")

# Load and scale images
car_image = pygame.image.load('object/rearcar.png')  # Make sure you have a 'rearcar.png' image in the same directory
car_image = pygame.transform.scale(car_image, (290, 270))  # Scale the car image to 290x270 pixels
car_rect = car_image.get_rect()
car_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 150)

tree_image = pygame.image.load('object/cactus.png')  # Make sure you have a 'tree.png' image in the same directory
tree_image = pygame.transform.scale(tree_image, (120, 150))  # Scale the tree image to 120x150 pixels

cloud_images = [
    #pygame.transform.scale(pygame.image.load('cloud1.png'), (100, 60)),
    pygame.transform.scale(pygame.image.load('object/cloud2.png'), (150, 90))
]

clouds = []
for _ in range(5):  # Create 5 clouds
    cloud = {
        'image': random.choice(cloud_images),
        'pos': [random.randint(0, SCREEN_WIDTH), random.randint(0, 5)],
        'speed': random.randint(1, 3)
    }
    clouds.append(cloud)

# Car speed settings
car_speed = 20
car_max_speed = 120
car_min_speed = 20
car_super_speed = 150

# Initialize road_y
road_y = 0

# Initialize Pygame font module
pygame.font.init()
font = pygame.font.SysFont(None, 36)

# Function to draw the road with perspective
def draw_road(screen, top_width, bottom_width, horizon_y, bottom_y, border_width):
    # Draw road borders
    top_border_width = border_width * (top_width / bottom_width)
    pygame.draw.polygon(screen, WHITE, [
        ((SCREEN_WIDTH - top_width) // 2 - top_border_width, horizon_y),  # Top-left border corner
        ((SCREEN_WIDTH - top_width) // 2, horizon_y),  # Top-left corner of the road
        ((SCREEN_WIDTH - bottom_width) // 2, bottom_y),  # Bottom-left corner of the road
        ((SCREEN_WIDTH - bottom_width) // 2 - border_width, bottom_y)  # Bottom-left border corner
    ])
    pygame.draw.polygon(screen, WHITE, [
        ((SCREEN_WIDTH + top_width) // 2, horizon_y),  # Top-right corner of the road
        ((SCREEN_WIDTH + top_width) // 2 + top_border_width, horizon_y),  # Top-right border corner
        ((SCREEN_WIDTH + bottom_width) // 2 + border_width, bottom_y),  # Bottom-right border corner
        ((SCREEN_WIDTH + bottom_width) // 2, bottom_y)  # Bottom-right corner of the road
    ])

    # Draw road
    pygame.draw.polygon(screen, GRAY, [
        ((SCREEN_WIDTH - top_width) // 2, horizon_y),  # Top-left corner of the road
        ((SCREEN_WIDTH + top_width) // 2, horizon_y),  # Top-right corner of the road
        ((SCREEN_WIDTH + bottom_width) // 2, bottom_y),  # Bottom-right corner of the road
        ((SCREEN_WIDTH - bottom_width) // 2, bottom_y)  # Bottom-left corner of the road
    ])
    pygame.draw.rect(screen, LIGHT_BLUE, (0, 0, SCREEN_WIDTH, horizon_y))  # Light blue sky above the road

# Function to draw trees parallel to the road
def draw_trees(screen, tree_image, road_bottom_width, horizon_y, bottom_y, road_y):
    num_trees = 15  # Number of trees on each side
    tree_distance = 200  # Distance between trees
    for i in range(num_trees):
        tree_y = (horizon_y - i * tree_distance + road_y) % SCREEN_HEIGHT
        if tree_y > horizon_y and tree_y < bottom_y:
            # Calculate the x position of trees based on road width at their y position
            road_width_at_y = road_bottom_width - ((road_bottom_width - 100) * (bottom_y - tree_y) / (bottom_y - horizon_y))
            left_tree_x = (SCREEN_WIDTH - road_width_at_y) // 2 - 280
            right_tree_x = (SCREEN_WIDTH + road_width_at_y) // 2 + 150
            screen.blit(tree_image, (left_tree_x, tree_y))
            screen.blit(tree_image, (right_tree_x, tree_y))

def draw_clouds(screen, clouds):
    for cloud in clouds:
        screen.blit(cloud['image'], cloud['pos'])
        cloud['pos'][0] -= cloud['speed']
        if cloud['pos'][0] < -cloud['image'].get_width():
            cloud['pos'][0] = SCREEN_WIDTH
            cloud['pos'][1] = random.randint(0, 5)

# Function to draw lane markings
def draw_lane_markings(screen, top_width, bottom_width, horizon_y, bottom_y, road_y):
    num_markings = 15  # Number of lane markings
    marking_distance = 200  # Distance between lane markings

    for i in range(num_markings):
        marking_y = (horizon_y + i * marking_distance + road_y) % SCREEN_HEIGHT
        if marking_y > horizon_y and marking_y < bottom_y:
            # Calculate the width and height of lane markings based on their y position
            perspective_factor = (marking_y - horizon_y) / (bottom_y - horizon_y)
            marking_width = int(10 + 20 * perspective_factor)
            marking_length = int(40 + 60 * perspective_factor)
            road_width_at_y = top_width + ((bottom_width - top_width) * perspective_factor)
            marking_x = (SCREEN_WIDTH - road_width_at_y) // 2 + road_width_at_y // 2 - marking_width // 2
            pygame.draw.rect(screen, WHITE, (marking_x, marking_y, marking_width, marking_length))

# Function to set car speed based on traffic sign class number
def set_car_speed(classNo):
    global car_speed
    # Speed limits based on traffic sign class numbers
    speed_limits = {
        0: 20,   # Speed Limit 20 km/h
        1: 30,   # Speed Limit 30 km/h
        2: 50,   # Speed Limit 50 km/h
        3: 60,   # Speed Limit 60 km/h
        4: 70,   # Speed Limit 70 km/h
        5: 80,   # Speed Limit 80 km/h
        6: 80,   # End of Speed Limit 80 km/h (maintain 80 km/h until a new limit is detected)
        7: 100,  # Speed Limit 100 km/h
        8: 120,  # Speed Limit 120 km/h
        9: car_speed,  # No passing
        10: car_speed, # No passing for vehicles over 3.5 metric tons
        11: car_speed, # Right-of-way at the next intersection
        12: car_speed, # Priority road
        13: car_speed, # Yield -
        14: 0, # Stop
        15: car_speed, # No vehicles -
        16: car_speed, # Vehicles over 3.5 metric tons prohibited
        17: 0, # No entry
        18: car_min_speed, # General caution
        19: car_min_speed, # Dangerous curve to the left
        20: car_min_speed, # Dangerous curve to the right
        21: car_speed, # Double curve
        22: car_min_speed, # Bumpy road
        23: car_min_speed, # Slippery road -
        24: car_min_speed, # Road narrows on the right
        25: car_min_speed, # Road work
        26: car_speed, # Traffic signals
        27: car_min_speed, # Pedestrians
        28: car_min_speed, # Children crossing
        29: car_min_speed, # Bicycles crossing
        30: car_speed, # Beware of ice/snow
        31: car_min_speed, # Wild animals crossing
        32: car_speed, # End of all speed and passing limits
        33: car_min_speed, # Turn right ahead
        34: car_min_speed, # Turn left ahead
        35: car_speed, # Ahead only
        36: car_speed, # Go straight or right
        37: car_speed, # Go straight or left
        38: car_speed, # Keep right
        39: car_speed, # Keep left
        40: car_speed, # Roundabout mandatory
        41: car_speed, # End of no passing
        42: car_speed  # End of no passing by vehicles over 3.5 metric tons
    }

    if classNo in speed_limits:
        car_speed = speed_limits[classNo]
    else:
        car_speed = car_min_speed  # Default speed if no specific speed limit



# Function to get traffic sign description
def get_traffic_sign_description(classNo):
    traffic_signs = {
        0: 'Speed Limit 20 km/h',
        1: 'Speed Limit 30 km/h',
        2: 'Speed Limit 50 km/h',
        3: 'Speed Limit 60 km/h',
        4: 'Speed Limit 70 km/h',
        5: 'Speed Limit 80 km/h',
        6: 'End of Speed Limit 80 km/h',
        7: 'Speed Limit 100 km/h',
        8: 'Speed Limit 120 km/h',
        9: 'No passing',
        10: 'No passing for vehicles over 3.5 metric tons',
        11: 'Right-of-way at the next intersection',
        12: 'Priority road',
        13: 'Yield',
        14: 'Stop',
        15: 'No vehicles',
        16: 'Vehicles over 3.5 metric tons prohibited',
        17: 'No entry',
        18: 'General caution',
        19: 'Dangerous curve to the left',
        20: 'Dangerous curve to the right',
        21: 'Double curve',
        22: 'Bumpy road',
        23: 'Slippery road',
        24: 'Road narrows on the right',
        25: 'Road work',
        26: 'Traffic signals',
        27: 'Pedestrians',
        28: 'Children crossing',
        29: 'Bicycles crossing',
        30: 'Beware of ice/snow',
        31: 'Wild animals crossing',
        32: 'End of all speed and passing limits',
        33: 'Turn right ahead',
        34: 'Turn left ahead',
        35: 'Ahead only',
        36: 'Go straight or right',
        37: 'Go straight or left',
        38: 'Keep right',
        39: 'Keep left',
        40: 'Roundabout mandatory',
        41: 'End of no passing',
        42: 'End of no passing by vehicles over 3.5 metric tons'
    }
    return traffic_signs.get(classNo, 'Unknown sign')

# Main loop
running = True
detected_classNo = 0
classIndex = 0
probabilityValue = 0
def game_loop(q, lock):
    global car_speed
    global road_y
    global detected_classNo
    global classIndex
    global probabilityValue
    last_update_time = time.time()  # Time of the last update
    update_interval = 5
    while running:
        # current_time = time.time()  # Get the current time
        # last_update_time = current_time
        #detected_classNo = random.randint(0, 8)
        set_car_speed(classIndex)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    car_speed = car_max_speed
                elif event.key == pygame.K_DOWN:
                    car_speed = car_min_speed
                elif event.key == pygame.K_SPACE:
                    car_speed = car_super_speed
                elif event.key == pygame.K_s:
                    car_speed = 0

        # Update the road and tree positions
        road_speed = car_speed
        road_y = (road_y + road_speed) % SCREEN_HEIGHT

        # Example: Set car speed based on a detected traffic sign
        # detected_classNo = random.randint(0, 42)  # Replace this with actual detection logic
        # set_car_speed(detected_classNo)

        # Draw everything
        screen.fill(BLUE)  # Draw main blue background

        # Draw grass
        pygame.draw.rect(screen, GREEN_YELLOW, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))

        # Draw road and sky
        road_top_width = 50  # Width of the road at the horizon
        road_bottom_width = 800  # Width of the road at the bottom
        border_width = 25
        horizon_y = 170  # Y-coordinate of the horizon (higher up)
        road_bottom_y = SCREEN_HEIGHT

        draw_road(screen, road_top_width, road_bottom_width, horizon_y, road_bottom_y, border_width)
        draw_lane_markings(screen, road_top_width, road_bottom_width, horizon_y, road_bottom_y, road_y)
        draw_trees(screen, tree_image, road_bottom_width, horizon_y, road_bottom_y, road_y)
        draw_clouds(screen, clouds)

        # Add shaking effect to the car
        shake_offset_y = random.randint(-1, 1)
        car_rect_with_shake = car_rect.move(0, shake_offset_y)

        # Draw car with shaking effect
        screen.blit(car_image, car_rect_with_shake)

        with lock:
            if not q.empty():
                classIndex, probabilityValue = q.get()

        # Render and display the current speed and traffic sign
        speed_text = f"Current Speed: {car_speed} km/h"
        speed_surface = font.render(speed_text, True, TEXT_COLOR)
        screen.blit(speed_surface, (10, 520))  # Position text at (10, 10)
        if probabilityValue > threshold:
            if classIndex in sign_images:
                sign_image_path = sign_images[classIndex]
                sign_image = pygame.image.load(sign_image_path)
                sign_image = pygame.transform.scale(sign_image, (100, 100))
            probability_text = f"Probability: {round((probabilityValue * 100), 2)}%"
            sign_description = get_traffic_sign_description(classIndex)
            sign_surface = font.render(sign_description, True, TEXT_COLOR)
            probability_surface = font.render(probability_text, True, TEXT_COLOR)
            screen.blit(sign_surface, (10, 10))  # Position text below speed text
            screen.blit(probability_surface, (10, 50))  # Position text below speed text
            screen.blit(sign_image, (680, 10))
        # Update the display
        pygame.display.flip()

        # Frame rate
        pygame.time.Clock().tick(30)

def camera_thread(q, lock):
    global classIndex
    global probabilityValue
    new_width = 250
    new_height = 200
    count = 0
    while True:
        # READ IMAGE
        success, imgOriginal = cap.read()

        if not success:
            break

        # PROCESS IMAGE
        img = np.asarray(imgOriginal)
        img = cv2.resize(img, (32, 32))
        img = preprocessing(img)
        img = img.reshape(1, 32, 32, 1)

        # PREDICT IMAGE
        predictions = model.predict(img)
        classIndex = np.argmax(predictions)
        probabilityValue = np.amax(predictions)

        # if probabilityValue > threshold:
        #     try:
        #         cv2.putText(imgOriginal, f"CLASS: {classIndex} {getClassName(classIndex)}", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        #         cv2.putText(imgOriginal, f"PROBABILITY: {round(probabilityValue * 100, 2)}%", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        #     except:
        #         print("TRY")
        with lock:
            if not q.full():
                q.put((classIndex, probabilityValue))
        # DISPLAY IMAGE
        resized_image = cv2.resize(imgOriginal, (new_width, new_height))
        cv2.imshow("Result", resized_image)

        # EXIT ON 'q' KEY PRESS
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(1 / 16)
    cap.release()
    cv2.destroyAllWindows()


q = queue.Queue(maxsize=10)
lock = threading.Lock()

# Create and start threads
camera_thread = threading.Thread(target=camera_thread, args=(q, lock))
game_thread = threading.Thread(target=game_loop, args=(q, lock))
camera_thread.start()
game_thread.start()

# Wait for threads to finish
camera_thread.join()
game_thread.join()

pygame.quit()
