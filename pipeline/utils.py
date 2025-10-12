import os
import numpy as np
import tempfile
from moviepy.video.io.VideoFileClip import VideoFileClip

class GetResponseClip:
    def __init__(self, retrieved_clips, n):
        self.res = retrieved_clips
        self.top_n_clips = n

    def get_clipdetails_from_rag(self):
        start_time, end_time, video_path = [], [], []
        metadata_list = self.res['metadatas'][0]  # list of metadata dicts
        distances = np.array(self.res['distances'][0])
        sorted_indices = np.argsort(distances)  # ascending order
 
        for counter in range(min(self.top_n_clips, len(sorted_indices))):
            idx = sorted_indices[counter]       # get the index of the clip
            metadata_dict = metadata_list[idx]
 
            start_time.append(metadata_dict['start_time'])
            end_time.append(metadata_dict['end_time'])
            video_path.append(metadata_dict['video_name'])
 
        # Return as dictionary to match usage in extract_video_from_metadata
        return {
            "start_time": start_time,
            "end_time": end_time,
            "video_path": video_path
        }

    def extract_video_from_metadata(self, output_folder=None):
        clipped_video_paths = []
        try:
            clip_details = self.get_clipdetails_from_rag()
            start_time = clip_details["start_time"]
            end_time = clip_details["end_time"]
            video_path = clip_details["video_path"]
 
            root_video_folder_path = r"C:\Users\Anil.Bhallavi\Desktop\work\Data Science\VideoRAG\dataset"
 
            # Default local folder for saving clips
            if output_folder is None:
                output_folder = os.path.join(root_video_folder_path, "clipped_videos")
            os.makedirs(output_folder, exist_ok=True)
 
            for idx in range(len(start_time)):
                final_video_path = os.path.join(root_video_folder_path, video_path[idx])
                if os.path.exists(final_video_path):
                    video = VideoFileClip(final_video_path)
                    # Use subclip instead of subclipped
                    clip = video.subclipped(start_time[idx], end_time[idx])
                    local_file_path = os.path.join(output_folder, f"clip_{idx}.mp4")
                    clip.write_videofile(local_file_path)
                   
                    # Close video objects to free resources
                    clip.close()
                    video.close()
                   
                    clipped_video_paths.append(local_file_path)
                else:
                    print(f"Video file does not exist: {final_video_path}")
 
            return clipped_video_paths
 
        except Exception as e:
            print(f"Error in getting video -> {e}")
            return []
 

# if __name__ == "__main__":
#     clipper = GetResponseClip(res, 2)
#     clipper.extract_video_from_metadata() 