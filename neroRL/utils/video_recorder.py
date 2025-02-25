import cv2
import os
import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ruamel
from jinja2 import Environment, FileSystemLoader
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

class VideoRecorder:
    """The VideoRecorder can be used to capture videos of the agent's behavior using enjoy.py or eval.py.
    The debug frame has been removed in this version so that only the environment and basic info are rendered."""
    def __init__(self, video_path, frame_rate):
        """Instantiates the VideoRecorder and initializes some members that affect the rendering of the video.
        
        Arguments:
            video_path {string} -- Path and filename for saving the to be recorded video.
            frame_rate {int} -- The frame rate of the to be rendered video.
        """
        self.font_face = cv2.FONT_HERSHEY_SIMPLEX
        self.scale = 0.4
        self.thickness = cv2.FILLED
        self.text_color = (255, 255, 255)
        self.margin = 2
        self.width = 420                                # Video dimensions
        self.height = 420
        self.info_height = 40
        self.video_path = video_path
        self.cwd = os.path.dirname(os.path.abspath(__file__))
        self.cwd = self.cwd[:self.cwd.rfind("neroRL") - 1]  # Fixed relative path
        self.website_path = self.cwd + "/result/"
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # Video codec
        self.frame_rate = int(frame_rate)

    def process_frame(self, frame_info):
        i, trajectory_data, width, height, info_height = frame_info
        # Process the environment frame
        env_frame = trajectory_data["vis_obs"][i][..., ::-1].astype(np.uint8)
        env_frame = cv2.resize(env_frame, (width, height), interpolation=cv2.INTER_AREA)
        
        # Create the info frame (same width as env_frame now)
        info_frame = np.zeros((info_height, width, 3), dtype=np.uint8)
        self.draw_text_overlay(info_frame, 8, 20, trajectory_data["seed"], "seed")
        self.draw_text_overlay(info_frame, 108, 20, i, "step")
        self.draw_text_overlay(info_frame, 208, 20, round(sum(trajectory_data["rewards"][0:i]), 3), "total reward")
        
        # Optionally overlay ground truth on the env_frame if available
        if "estimated_ground_truth" in trajectory_data and len(trajectory_data["estimated_ground_truth"]) > 0:
            point_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
            for j in range(0, len(trajectory_data["estimated_ground_truth"][i]), 2):
                x = trajectory_data["estimated_ground_truth"][i][j].clip(0, 1)
                y = trajectory_data["estimated_ground_truth"][i][j + 1].clip(0, 1)
                position = (int(x * width), int(y * height))
                point_color = point_colors[j // 2]
                point_radius = 8
                cv2.circle(env_frame, position, point_radius, point_color, -1)
                
        if len(env_frame.shape) == 2:
            env_frame = cv2.cvtColor(env_frame, cv2.COLOR_GRAY2BGR)
            
        # Stack info frame on top of the environment frame
        output_image = np.vstack((info_frame, env_frame))
        return output_image

    def render_video(self, trajectory_data):
        # Adjust the video writer size: now width is self.width (not self.width*2)
        out = cv2.VideoWriter(
            self.video_path + "_seed_" + str(trajectory_data["seed"]) + ".mp4",
            self.fourcc, self.frame_rate, (self.width, self.height + self.info_height)
        )

        total_frames = len(trajectory_data["vis_obs"])
        for i in tqdm(range(total_frames), desc="Processing frames"):
            frame_info = (i, trajectory_data, self.width, self.height, self.info_height)
            frame = self.process_frame(frame_info)
            out.write(frame)
        out.release()

    def _config_to_html(self, config, prfx=""):
        """Returns a html string that contains the configuration of the key
        
        Arguments:
            config {dict} -- The configuration
        
        Returns:
            {string}  -- The html string that contains the configuration of the key.
        """
        tab = "&nbsp;&nbsp;&nbsp;&nbsp;" * 2
        html = ""
        for key in config:
            if type(config[key]) is ruamel.yaml.comments.CommentedMap:
                html += prfx + "<b>" + str(key) + "</b>: " + "<br>" + self._config_to_html(config[key], prfx + tab)
            else:
                html += prfx + "<b>" + str(key) + "</b>: " + str(config[key]) + "<br>"
        return html

    def generate_website(self, trajectory_data, configs):
        """Generates a website that can be used to view the trajectory data.
        
        Arguments:
            trajectory_data {dict} -- This dictionary provides all the necessary information to render a website.
            config {dict} -- The configuration
        """
        # Create the video path for the website if it does not exist
        video_path = self.website_path + "videos/"
        if not os.path.exists(video_path):
            os.makedirs(video_path)
            
        # Create an id for the website
        id = self._generate_id()
        # Generate the videos for the website
        video_paths = self._generate_website_videos(trajectory_data, video_path)
        
        # Prepare the data for the website
        action_probs = []
        for probs in trajectory_data["probs"]:
            action_probs.append([action_branch.squeeze(dim=0).tolist() for action_branch in probs])
        
        action_names, actions = trajectory_data["action_names"], trajectory_data["actions"]
        values, entropies = np.array(trajectory_data["values"]).tolist(), trajectory_data["entropies"]
        
        env_info = self._config_to_html(configs["environment"])
        model_info = self._config_to_html(configs["model"])
        hyper_info = self._config_to_html(configs["trainer"])
        sampler_info = self._config_to_html(configs["sampler"])
        
        # Load the template file
        template_env = Environment(loader=FileSystemLoader(searchpath=self.website_path))
        template = template_env.get_template("./template/result_website.html")  
        
        # Render the template
        with open(self.website_path + 'result_website_' + str(id) + '.html', 'w') as output_file:
            output_file.write(template.render(
                envInfo=env_info,
                hyperInfo=hyper_info,
                modelInfo=model_info,
                samplerInfo=sampler_info,
                videoPath=str(video_paths),
                yValues=str(values),
                yEntropy=str(entropies),
                yAttentionWeights=str(trajectory_data["attention_weights"]),
                yAction=str(action_probs),
                action=str(actions),
                actionNames=str(action_names) if action_names is not None else "null",
                frameRate=str(self.frame_rate)
            ))
            
    def _generate_website_videos(self, trajectory_data, video_path):
        """Generates the videos for the website.

        Arguments:
            trajectory_data {dict} -- This dictionary provides all the necessary information to render a website.
            video_path {string} -- The path where the videos should be saved.

        Returns:
            {list} -- A list of video paths.
        """
        video_paths = []
        for key in ["vis_obs", "decoder_frames", "agent_frames"]:
            if len(trajectory_data[key]) > 0:
                # Generate an id
                id = self._generate_id()
                # Render the trajectory data to a video
                self._render_environment_episode(key, trajectory_data, video_path, str(id))
                # Add the video path to the list
                video_paths.append("videos/video_seed_" + str(trajectory_data["seed"]) + "_" + str(id) + ".webm")
                # Render the trajectory data of gt to a video
                if len(trajectory_data["estimated_ground_truth"]) > 0:
                    gt_id = self._generate_id()
                    self._render_environment_episode(key, trajectory_data, video_path, str(gt_id), True)
                    video_paths.append("videos/video_seed_" + str(trajectory_data["seed"]) + "_" + str(gt_id) + ".webm")
                else:
                    video_paths.append("")
            else:
                video_paths.append("")
                video_paths.append("")
            
        return video_paths
        
    def _generate_id(self):
        """Generates a unique id.
        
        Returns:
            {string} -- The unique id.
        """
        result_website_names, video_names = os.listdir(self.website_path), os.listdir(self.website_path + "videos/")
        file_names = result_website_names + video_names
        
        ids = [0]
        for file_name in file_names:
            file_name_prx = file_name.split(".")[0].split("_")[-1]
            if re.match(r'\d+$', file_name_prx):
                ids.append(int(file_name_prx))

        id = max(ids) + 1
        return str(id)
    
    def _render_environment_episode(self, key, trajectory_data, path, video_id, gt=False):
        """Renders an episode of an agent behaving in its environment.
        
        Arguments:
            trajectory_data {dict} -- This dictionary provides all the necessary information to render one episode of an agent behaving in its environment.
            video_id {string} -- The id of the video.
            gt {bool} -- If true the estimated ground truth is rendered.
        """
        # Set fourcc s.t. the video is saved as webm
        webm_fourcc = cv2.VideoWriter_fourcc(*'VP09')
        
        # Init VideoWriter, the frame rate is defined by each environment individually
        out = cv2.VideoWriter(
            path + "video_seed_" + str(trajectory_data["seed"]) + "_" + video_id + ".webm",
            webm_fourcc, 1, (self.width, self.height + self.info_height)
        )
        
        for i in range(len(trajectory_data[key])):
            # Setup environment frame
            env_frame = trajectory_data[key][i][..., ::-1].astype(np.uint8)  # Convert RGB to BGR
            env_frame = cv2.resize(env_frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

            # Setup info frame
            info_frame = np.zeros((self.info_height, self.width, 3), dtype=np.uint8)
            # Seed
            self.draw_text_overlay(info_frame, 8, 20, trajectory_data["seed"], "seed")
            # Current step
            self.draw_text_overlay(info_frame, 108, 20, i, "step")
            # Collected rewards so far
            self.draw_text_overlay(info_frame, 208, 20, round(sum(trajectory_data["rewards"][0:i]), 3), "total reward")

            if i == len(trajectory_data[key]) - 1:
                self.draw_text_overlay(info_frame, 368, 20, "True", "episode done")
            else:
                self.draw_text_overlay(info_frame, 368, 20, "False", "episode done")
                
            # Plot estimated ground truth if requested
            if gt:
                point_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
                for j in range(0, len(trajectory_data["estimated_ground_truth"][i]), 2):
                    x = trajectory_data["estimated_ground_truth"][i][j].clip(0, 1)
                    y = trajectory_data["estimated_ground_truth"][i][j + 1].clip(0, 1)
                    position = (int(x * self.width), int(y * self.height))
                    point_color = point_colors[j // 2]
                    point_radius = 8
                    cv2.circle(env_frame, position, point_radius, point_color, -1)
            
            # Concatenate info and environment frames
            output_image = np.vstack((info_frame, env_frame))
            out.write(output_image)
        out.release()
        
    def draw_text_overlay(self, frame, x, y, value, label):
        """Draws text on a frame at some position to display a value and its associated label.
        The text will look like "label: value".
        
        Arguments:
            frame {nd.array} -- The to be edited frame
            x {int} -- Starting point of the text on the X-Axis
            y {int} -- Starting point of the text on the Y-Axis
            value {float} -- The to be rendered value
            label {string} -- The associated label of the value
        """
        bg_color = (0, 0, 0)
        pos = (x, y)
        text = label + ": " + str(value)
        txt_size = cv2.getTextSize(text, self.font_face, self.scale, self.thickness)
        end_x = pos[0] + txt_size[0][0] + self.margin
        end_y = pos[1] - txt_size[0][1] - self.margin
        cv2.rectangle(frame, pos, (end_x, end_y), bg_color, self.thickness)
        cv2.putText(frame, text, pos, self.font_face, self.scale, self.text_color, 1, cv2.LINE_AA)

    def draw_bar(self, frame, x, y, prob, label, chosen):
        """Draws bars and text on a frame at some position to display an actions probability.
        It will be colored green if that action was selected by the agent.
        Otherwise it will be orange.
        
        Arguments:
            frame {nd.array} -- The to be edited frame
            x {int} -- Starting point of the text on the X-Axis
            y {int} -- Starting point of the text on the Y-Axis
            prob {float} -- The probability of the concerned action
            label {string} -- The associated label of the value
            chosen {bool} -- Whether the action was selected by the agent
        """
        if chosen:
            bg_color = (0, 255, 0)
        else:
            bg_color = (0, 69, 255)
            
        pos = (x, y)
        text = label + ": " + str(prob)
        txt_size = cv2.getTextSize(text, self.font_face, self.scale, self.thickness)
        end_x = int(self.width * prob)
        end_y = pos[1] - txt_size[0][1] - self.margin
        cv2.rectangle(frame, pos, (end_x, end_y), bg_color, self.thickness)
        cv2.putText(frame, text, (x + 5, y), self.font_face, self.scale, self.text_color, 1, cv2.LINE_AA)

    @staticmethod
    def line_plot(data: np.ndarray, label: str, marker_pos=10) -> np.ndarray:
        matplotlib.use('Agg')
        font = {"weight": "bold", "size": 22}
        matplotlib.rc('font', **font)
        plt.style.use("dark_background")
        fig = plt.figure(dpi=180)
        fig.set_size_inches(14, 6)
        ax = fig.subplots()
        ax.plot([marker_pos], data[marker_pos], fillstyle="full", markersize=12, marker="o", color="r")
        x = list(range(len(data)))
        ax.plot(x, data)
        ax.set_title("Value: " + str(data[marker_pos]))
        ax.set_xlim([0, len(data)])
        ax.set_xlabel("Episode Steps")
        ax.set_ylabel(label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        return fig
    
    @staticmethod
    def fig_to_ndarray(fig) -> np.ndarray:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig.tight_layout(pad=0)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        plt.close(fig)
        return np.asarray(buf)

    @staticmethod
    def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), int(height))
        else:
            r = width / float(w)
            dim = (int(width), int(h * r))
        resized = cv2.resize(image, dim, interpolation=inter)
        return resized
