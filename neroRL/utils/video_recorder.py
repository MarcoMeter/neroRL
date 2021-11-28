import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class VideoRecorder:
    """The VideoRecorder can be used to capture videos of the agent's behavior using enjoy.py or eval.py.
    Along with the agent's behavior a debug frame is rendered that shows information such as the action probabilities and the state's value."""
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
        self.website_path = video_path
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # Video codec
        self.frame_rate = int(frame_rate)

    def render_video(self, trajectory_data):
        """Triggers the process of rendering the trajectory data to a video.
        The rendering is done with the help of OpenCV.
        
        Arguments:
            trajectory_data {dift} -- This dictionary provides all the necessary information to render one episode of an agent behaving in its environment.
        """
        # Init VideoWriter, the frame rate is defined by each environment individually
        out = cv2.VideoWriter(self.video_path + "_seed_" + str(trajectory_data["seed"]) + ".mp4",
                                self.fourcc, self.frame_rate, (self.width * 2, self.height + self.info_height))
        for i in range(len(trajectory_data["vis_obs"])):
            # Setup environment frame
            env_frame = trajectory_data["vis_obs"][i][...,::-1].astype(np.uint8) # Convert RGB to BGR, OpenCV expects BGR
            env_frame = cv2.resize(env_frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

            # Setup info frame
            info_frame = np.zeros((self.info_height, self.width * 2, 3), dtype=np.uint8)
            # Seed
            self.draw_text_overlay(info_frame, 8, 20, trajectory_data["seed"], "seed")
            # Current step
            self.draw_text_overlay(info_frame, 108, 20, i, "step")
            # Collected rewards so far
            self.draw_text_overlay(info_frame, 208, 20, round(sum(trajectory_data["rewards"][0:i]), 3), "total reward")

            # Setup debug frame
            debug_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            if not i == len(trajectory_data["vis_obs"]) - 1:
                # Action probabilities
                next_y = 20
                for x, probs in enumerate(trajectory_data["log_probs"][i]):
                    self.draw_text_overlay(debug_frame, 5 , next_y, round(trajectory_data["entropies"][i][x], 5), "entropy dimension " + str(x))
                    next_y += 20
                    for y, prob in enumerate(probs.squeeze(dim=0)):
                        if trajectory_data["action_names"] is not None:
                            label = str(trajectory_data["action_names"][x][y])
                        else:
                            label = str(y)
                        self.draw_bar(debug_frame, 0, next_y, round(prob.item(), 10), label, y == trajectory_data["actions"][i][x])
                        next_y += 20
                    next_y += 10

                # Plot value
                # hard-coded next_y because if too many actions are available, this plot does not fit
                next_y = 230
                fig = VideoRecorder.line_plot(trajectory_data["values"], "value", marker_pos=i)
                img = VideoRecorder.fig_to_ndarray(fig)[:,:,0:3] # Drop Alpha
                img = VideoRecorder.image_resize(img, width=self.width, height=None)
                debug_frame[next_y : next_y + img.shape[0], 0 : img.shape[1], :] = img

            else:
                self.draw_text_overlay(debug_frame, 5, 60, "True", "episode done")

            # Concatenate environment and debug frames
            output_image = np.hstack((env_frame, debug_frame))
            output_image = np.vstack((info_frame, output_image))

            # Write frame
            out.write(output_image)
        # Finish up the video
        out.release()

    def generate_website(self, trajectory_data):
        """Generates a website that can be used to view the recorded video.
        
        Arguments:
            trajectory_data {dift} -- This dictionary provides all the necessary information to render one episode of an agent behaving in its environment.
        """
        webm_fourcc = cv2.VideoWriter_fourcc(*'VP09')
        
        # Init VideoWriter, the frame rate is defined by each environment individually
        out = cv2.VideoWriter(self.website_path + "_seed_" + str(trajectory_data["seed"]) + ".webm",
                                webm_fourcc, self.frame_rate, (self.width * 2, self.height + self.info_height))
        
        for i in range(len(trajectory_data["vis_obs"])):
            # Setup environment frame
            env_frame = trajectory_data["vis_obs"][i][...,::-1].astype(np.uint8) # Convert RGB to BGR, OpenCV expects BGR
            env_frame = cv2.resize(env_frame, (self.width * 2, self.height), interpolation=cv2.INTER_AREA)

            # Setup info frame
            info_frame = np.zeros((self.info_height, self.width * 2, 3), dtype=np.uint8)
            # Seed
            self.draw_text_overlay(info_frame, 8, 20, trajectory_data["seed"], "seed")
            # Current step
            self.draw_text_overlay(info_frame, 108, 20, i, "step")
            # Collected rewards so far
            self.draw_text_overlay(info_frame, 208, 20, round(sum(trajectory_data["rewards"][0:i]), 3), "total reward")

            # Concatenate environment and debug frames
            output_image = np.vstack((info_frame, env_frame))

            # Write frame
            out.write(output_image)
        # Finish up the video
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
    def line_plot(data: np.ndarray, label: str, marker_pos = 10) -> np.ndarray:
        matplotlib.use('Agg')
        font = {"weight" : "bold", "size" : 22}
        matplotlib.rc('font', **font)
        # Setup figure
        plt.style.use("dark_background")
        fig = plt.figure(dpi=180)
        fig.set_size_inches(14, 6)
        ax = fig.subplots()

        # Plot marker
        ax.plot([marker_pos], data[marker_pos], fillstyle="full", markersize=12, marker="o", color="r")
        x = [i for i in range(0, len(data))]

        # Line plot
        ax.plot(x, data)

        # Annotate marker
        # ax.annotate(str(data[marker_pos]), (x[marker_pos] + .011 ,data[marker_pos] + .011), color = "r")
        ax.set_title("Value: " + str(data[marker_pos]))

        # X and Y axis
        ax.set_xlim([0,len(data)])
        ax.set_xlabel("Episode Steps")
        ax.set_ylabel(label)

        # Remove borders
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Text color
        # ax.tick_params(color='gray', labelcolor='gray')
        # for spine in ax.spines.values():
        #     spine.set_edgecolor('gray')
        return fig
    
    @staticmethod
    def fig_to_ndarray(fig) -> np.ndarray:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        # Attach figure to canvas
        fig.tight_layout(pad=0)
        canvas = FigureCanvasAgg(fig)

        # Retrieve a view on the renderer buffer
        canvas.draw()

        buf = canvas.buffer_rgba()

        plt.close(fig)

        # Convert to a NumPy array
        return np.asarray(buf)

    @staticmethod
    def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), int(height))

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (int(width), int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation = inter)

        # return the resized image
        return resized