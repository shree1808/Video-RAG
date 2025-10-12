from moviepy.video.io.VideoFileClip import VideoFileClip
import os , re
from PIL import Image 

class ClipProcessor:
    def __init__(self, input_videos_path: str):
        self.input_folder = input_videos_path 
        self.clip_duration = 20  # 30 seconds
        self.frame_interval = 5 # capture frame every 5 sec
        self.folders = ['audio_from_clips', 'frames_from_clips']
        for folder in self.folders:
            os.makedirs(folder, exist_ok=True)  # create folders if not exist

    def get_videos_from_folder(self):
        video_files = []
        for file in os.listdir(self.input_folder):
            if file.endswith('.mp4'):   
                video_files.append(file)
        return video_files

    def get_clips(self, video_file):
        # Full path to video
        video_path = os.path.join(self.input_folder, video_file)
        original_video = VideoFileClip(video_path)
        match = re.match(r'(.+)\.mp4$', video_file)
        if match:
            filename = match.group(1)

        duration = original_video.duration 

        for start in range(0, int(duration), self.clip_duration):
            end_time = min(start + self.clip_duration, duration)
            try:
                video_clip = original_video.subclipped(start, end_time)
                audio = self.get_audio_from_clips(video_clip)

                # Save audio
                aud_path = os.path.join(self.folders[0], f"clip_{filename}_{start}_{end_time}.wav")
                audio.write_audiofile(aud_path)
                print(f"Saved audio: {aud_path}")

                # Save frames
                self.get_frames_from_clips(clip = video_clip, filename= filename, start= start, end_time= end_time)

            except Exception as e:
                print('Clipping/Audio Extraction failed with error:', e)

    def get_audio_from_clips(self, clip):
        return clip.audio

    def get_frames_from_clips(self, clip, filename, start, end_time):
        frame_times = range(0, int(clip.duration), self.frame_interval)
        for t in frame_times:
            try:
                frame = clip.get_frame(t)  # numpy array (H,W,3)
                img = Image.fromarray(frame)
                frame_path = os.path.join(
                    self.folders[1],
                    f"frame_{filename}_{start+t}_{start+t+self.frame_interval}.jpg"
                )
                img.save(frame_path)
                print(f"Saved frame: {frame_path}")
            except Exception as e:
                print(f"Frame extraction failed at {t}s: {e}")

    def main(self):
        print('Starting Clip Generation and Audio Extraction')
        print('='*40)
        video_files = self.get_videos_from_folder()
        for file in video_files:
            self.get_clips(file)
        print('='*40)
        print('Extraction Done!!')


# if __name__ == "__main__":
#     input_folder = r"C:\Users\shree.sudame\NV_POC_Projects\video_rag\video_folder"
#     processor = ClipProcessor(input_videos_path=input_folder)
#     processor.main()
