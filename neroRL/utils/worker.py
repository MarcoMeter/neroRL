import multiprocessing
import multiprocessing.connection

from neroRL.environments.wrapper import wrap_environment

def worker_process(remote: multiprocessing.connection.Connection, env_config, worker_id: int, record_video = False):
    """Initializes the environment and executes its interface.

    Arguments:
        remote {multiprocessing.connection.Connection} -- Parent thread
        env_config {dict} -- The configuration data of the desired environment
        worker_id {int} -- Id for the environment's process. This is necessary for Unity ML-Agents environments, because these operate on different ports.
    """

    # Initialize and wrap the environment
    try:
        env = wrap_environment(env_config, worker_id, record_trajectory = record_video)
    except KeyboardInterrupt:
        pass

    # Communication interface of the environment thread
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                remote.send(env.step(data))
            elif cmd == "reset":
                remote.send(env.reset(data))
            elif cmd == "close":
                remote.send(env.close())
                remote.close()
                break
            elif cmd == "video":
                remote.send(env.get_episode_trajectory)
            else:
                raise NotImplementedError
        except:
            break

class Worker:
    """A worker that runs one thread and controls its own environment instance."""
    child: multiprocessing.connection.Connection
    process: multiprocessing.Process
    
    def __init__(self, env_config, worker_id: int, record_video = False):
        """
        Arguments:
            env_config {dict -- The configuration data of the desired environment
            worker_id {int} -- worker_id {int} -- Id for the environment's process. This is necessary for Unity ML-Agents environments, because these operate on different ports.
        """
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process, args=(parent, env_config, worker_id, record_video))
        self.process.start()