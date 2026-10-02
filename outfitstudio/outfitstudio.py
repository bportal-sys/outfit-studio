import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO

class outfit_studio:
    def __init__(self, model = "yolov8n-pose.pt", folder_dir='outfit_output', clean_outfit_output = 'clean', label_outfit_output = 'labeled' ):
        self.base_dir = os.getcwd() 
        self.clean_dir = os.path.join(self.base_dir, folder_dir, clean_outfit_output)
        self.labeled_dir = os.path.join(self.base_dir, folder_dir, label_outfit_output)
        os.makedirs(self.clean_dir, exist_ok=True)
        os.makedirs(self.labeled_dir, exist_ok=True)

        self.pose_model = YOLO(model) 

    def main(self, default_input= 'apple_cont'):
        cap = self.find_device()
        print("\n Ready! Spacebar = Take Picture | ESC = Quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Lost camera stream connection.")
                break

            display_frame = self.pose_model_result(frame)
                    
            cv2.imshow("Outfit Capture Mode", display_frame)

            k = cv2.waitKey(1)
            if k % 256 == 27:  # ESC key
                print("Closing application...")
                break
            elif k % 256 == 32:  # Spacebar key
                print("📸 Action! Snapshot in 1 second...")
                time.sleep(1.0)
                        
                for _ in range(5):
                    cap.read()
                ret, fresh_frame = cap.read()
                        
                if ret and fresh_frame is not None:
                    self.save_pic(fresh_frame)
                    print(f" Success! Saved:")
                    print(f"   🧼 Clean (No marks): {self.clean_path}")
                    print(f"   🏷️ Labeled (Skeleton + Score): {self.labeled_path}")
                else:
                    print("❌ Failed to capture a fresh frame.")

        cap.release()
        cv2.destroyAllWindows()


    def find_device(self, default_input = 'apple_cont'):
        cap = None
        for index in range(5):
            test_cap = cv2.VideoCapture(index)
            if test_cap.isOpened():
                ret, frame = test_cap.read()
                if ret and frame is not None:
                    print(f" Found active camera stream at Index {index}!")
                    cap = test_cap
                    break
                test_cap.release()

        if cap is None:
            print("\n❌ All index streams came back black or closed.")
            exit()
        return cap

    def pose_model_result(self, frame):
        results = self.pose_model(frame, verbose=False)
        pose_score_text = "Pose Score: N/A"
            
        # Generate live preview frame with both skeleton overlay AND text score
        display_frame = frame.copy()
        for result in results:
            if result.keypoints is not None and len(result.keypoints.conf) > 0:
                # FIXED: results extract a 2D array [[conf1, conf2...]]. We select [0] for person 1.
                confidences = result.keypoints.conf.tolist()
                if confidences and len(confidences[0]) > 0:
                    person_conf = confidences[0]
                    avg_conf = sum(person_conf) / len(person_conf)
                    pose_score_text = f"Pose Score: {int(avg_conf * 100)}%"
                
                # Draw the real-time visual skeleton onto the preview
                display_frame = result.plot() 

        h, w, _ = display_frame.shape
        cv2.putText(display_frame, pose_score_text, (w - 260, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        return display_frame

        
    def save_pic(self, fresh_frame):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"outfit_{timestamp}.png"
                    
        # 1. Save pure un-mapped frame to /clean/
        self.clean_path = os.path.join(self.clean_dir, filename)
        cv2.imwrite(self.clean_path, fresh_frame)
                    
        # 2. Process final shot for the /labeled/ version
        fresh_results = self.pose_model(fresh_frame, verbose=False)
        final_score_text = "Pose Score: N/A"
                    
        # Start labeled image with a clean copy
        labeled_frame = fresh_frame.copy()
                    
        for r in fresh_results:
            if r.keypoints is not None and len(r.keypoints.conf) > 0:
                confs = r.keypoints.conf.tolist()
                # FIXED: Apply the index correction here too for saving
                if confs and len(confs[0]) > 0:
                    person_confs = confs[0]
                    avg_conf = sum(person_confs) / len(person_confs)
                    final_score_text = f"Pose Score: {int(avg_conf * 100)}%"
                            
                # Maps out the physical skeleton dots and limbs onto the frame array
                labeled_frame = r.plot()
        
        h, w, _ = labeled_frame.shape
        # Burn the numeric text score on top of the skeleton drawing
        cv2.putText(labeled_frame, final_score_text, (w - 260, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        # Save fully mapped frame to /labeled/
        self.labeled_path = os.path.join(self.labeled_dir, filename)
        cv2.imwrite(self.labeled_path, labeled_frame)
                    
if __name__ == "__main__":
    studio = outfit_studio()
    studio.main()