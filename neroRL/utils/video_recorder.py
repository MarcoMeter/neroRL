import cv2
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
        self.video_path = video_path
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
                                self.fourcc, self.frame_rate, (self.width * 2, self.height))
        for i in range(len(trajectory_data["vis_obs"])):
            # Setup environment frame
            env_frame = trajectory_data["vis_obs"][i][...,::-1].astype(np.uint8) # Convert RGB to BGR, OpenCV expects BGR
            env_frame = cv2.resize(env_frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

            # Setup debug frame
            debug_frame = np.zeros((420, 420, 3), dtype=np.uint8)
            # Current step
            self.draw_text_overlay(debug_frame, 5, 20, i, "step")
            # Collected rewards so far
            self.draw_text_overlay(debug_frame, 215, 20, round(trajectory_data["rewards"][i], 3), "total reward")
            if not i == len(trajectory_data["vis_obs"]) - 1:
                # Current value of the state
                self.draw_text_overlay(debug_frame, 5, 40, round(trajectory_data["values"][i].item(), 5), "value")
                # Current entropy
                self.draw_text_overlay(debug_frame, 215, 40, sum(trajectory_data["entropies"][i]), "entropy")
                # Environment seed
                self.draw_text_overlay(debug_frame, 215, 60, trajectory_data["seed"], "seed")
                # Selected action
                for index, action in enumerate(trajectory_data["actions"][i]):
                    if trajectory_data["action_names"] is not None:
                        action_name = trajectory_data["action_names"][index][action]
                    else:
                        action_name = ""
                    self.draw_text_overlay(debug_frame, 5 + (210 * (index % 2)), 60 + (20 * (int(index / 2))),
                                            str(action) + " " + action_name, "action " + str(index))
                # Action probabilities
                next_y = 100
                for x, probs in enumerate(trajectory_data["log_probs"][i]):
                    self.draw_text_overlay(debug_frame, 5 , next_y, round(trajectory_data["entropies"][i][x], 5), "entropy dim" + str(x))
                    next_y += 20
                    for y, prob in enumerate(probs.squeeze(dim=0)):
                        if trajectory_data["action_names"] is not None:
                            label = str(trajectory_data["action_names"][x][y])
                        else:
                            label = ""
                        self.draw_bar(debug_frame, 0, next_y, round(prob.item(), 10), label, y == trajectory_data["actions"][i][x])
                        next_y += 20
                    next_y += 10
            else:
                self.draw_text_overlay(debug_frame, 5, 60, "True", "episode done")

            # Concatenate environment and debug frames
            output_image = np.hstack((env_frame, debug_frame))

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
        end_x = int(420 * prob)
        end_y = pos[1] - txt_size[0][1] - self.margin
        cv2.rectangle(frame, pos, (end_x, end_y), bg_color, self.thickness)
        cv2.putText(frame, text, (x + 5, y), self.font_face, self.scale, self.text_color, 1, cv2.LINE_AA)
