import sys
import os
import argparse
import torch
from torchvision import transforms as trans
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from utils.util import *
from utils.align_trans import *
from src.detector import detect_faces
from face_model import MobileFaceNet, l2_norm
from facebank import load_facebank, prepare_facebank
import cv2
import time
# from PIL import Image

def resize_image(img, scale):
    """
        resize image
    """
    height, width, channel = img.shape
    new_height = int(height * scale)     # resized new height
    new_width = int(width * scale)       # resized new width
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)      # resized image
    return img_resized

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='face detection demo')
    parser.add_argument('-th','--threshold',help='threshold score to decide identical faces',default=70, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true", default= False)
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true", default= False)
    parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true",default= True )
    parser.add_argument("--scale", dest='scale', help="input frame scale to accurate the speed", default=0.5, type=float)
    parser.add_argument('--mini_face', dest='mini_face', help=
    "Minimum face to be detected. derease to increase accuracy. Increase to increase speed",
                        default=10, type=int)
    # Checing my computer has GPU,.. 
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load model the start.
    detect_model = MobileFaceNet(512).to(device)  # embeding size is 512 (feature vector)
    detect_model.load_state_dict(torch.load('Weights/MobileFace_Net', map_location=lambda storage, loc: storage))
    print('MobileFaceNet face detection model generated')
    detect_model.eval()
    # Loading data to preivous of the system.
    if args.update:
        targets, names = prepare_facebank(detect_model, path='facebank', tta=args.tta)
        print('facebank updated')
    else:
        targets, names = load_facebank(path='facebank')
        print('facebank loaded')
        # targets: number of candidate x 512


    n = 0                        ###########
    dem_sai = 0                  ###########

    cap = cv2.VideoCapture('test.mp4')
    while True:
        isSuccess, frame = cap.read()
        
        if isSuccess:
            # Convert color for the frames
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                start_time = time.time()
                # Using PIL package to  convert frame to array's PLI
                imgs = Image.fromarray(frame)
                bboxes, landmarks = detect_faces(imgs, min_face_size=200.0)
                # Tranfor the landmark to the  simarler poiter
                faces = Face_alignment(frame, default_square=True, landmarks=landmarks)
                embs = []
                test_transform = trans.Compose([
                                trans.ToTensor(),
                                trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
                
                for img in faces:
                    if args.tta:
                        mirror = cv2.flip(img,1)
                        emb = detect_model(test_transform(img).to(device).unsqueeze(0))
                        emb_mirror = detect_model(test_transform(mirror).to(device).unsqueeze(0))
                        embs.append(l2_norm(emb + emb_mirror))
                    else:
                        embs.append(detect_model(test_transform(img).to(device).unsqueeze(0)))
                # test embs face for system.
                source_embs = torch.cat(embs)  # number of detected faces x 512
                diff = source_embs.unsqueeze(-1) - targets.transpose(1, 0).unsqueeze(0) # i.e. 3 x 512 x 1 - 1 x 512 x 2 = 3 x 512 x 2
                dist = torch.sum(torch.pow(diff, 2), dim=1) # number of detected faces x numer of target faces
                minimum, min_idx = torch.min(dist, dim=1) # min and idx for each row
                print("min_idx-------------->",min_idx)

                temp = str(min_idx)

                if(temp != 'tensor([0])'):
                    dem_sai +=1
                    print("Dem sai = ",dem_sai)

                min_idx[minimum > ((args.threshold-156)/(-80))] = -1  # if no match, set idx to -1
                score = minimum
                results = min_idx
                

                # convert distance to score dis(0.7,1.2) to score(100,60)
                score_100 = torch.clamp(score*-80+156,0,100)

                image = Image.fromarray(frame)
                draw = ImageDraw.Draw(image)
                font = ImageFont.truetype('utils/simkai.ttf', 30)

                FPS = 1.0 / (time.time() - start_time)
                draw.text((10, 10), 'FPS: {:.1f}'.format(FPS), fill=(0, 0, 0), font=font)

                for i, b in enumerate(bboxes):
                    draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline='blue', width=5)
                    if args.score:

                        draw.text((int(b[0]), int(b[1]-25)), names[results[i] + 1] + ' score:{:.0f}'.format(score_100[i]), fill=(255,255,0), font=font)
                    else:
                        draw.text((int(b[0]), int(b[1]-25)), names[results[i] + 1], fill=(255,255,0), font=font)
                        print(names[results[i] + 1])

                for p in landmarks:
                    for i in range(5):
                        draw.ellipse([(p[i] - 2.0, p[i + 5] - 2.0), (p[i] + 2.0, p[i + 5] + 2.0)], outline='blue')
                frame = np.asarray(image)

                n=n+1           ##########
                print("N =",n)

            except:
                pass
                print('detect error')
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('video', frame)

        if n ==1000:            ##########
            print("Dem sai = ",dem_sai)
            break               ##########

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
