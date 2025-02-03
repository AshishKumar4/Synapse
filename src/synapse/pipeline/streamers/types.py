from queue import Queue
import threading

from synapse.utils import DataFrame
from synapse.pipeline.sources import DataSource, EventDrivenDataSource
from synapse.pipeline.sinks import DataSink

class DataStreamer(DataSource, DataSink):
    def __init__(self) -> None:
        super(DataStreamer, self).__init__()
        self.msg_queue = Queue()
        self.is_closed = False
        pass
    
    def commit(self, data: DataFrame):
        self.msg_queue.put((data, False))
        
    def clear(self):
        """
        Clear any pending data in the message queue.
        This is done by repeatedly removing items from the queue until it is empty.
        """
        with self.msg_queue.mutex:
            self.msg_queue.queue.clear()
            # Reset the unfinished tasks count to zero so that join() doesn’t block forever
            self.msg_queue.unfinished_tasks = 0
            self.msg_queue.all_tasks_done.notify_all()
        
    def __next__(self) -> DataFrame:
        data, is_final = self.msg_queue.get()
        self.msg_queue.task_done()
        if is_final:
            raise StopIteration
        return data
    
    def __iter__(self) -> DataFrame:
        return self
    
    def read_from(self, data_source: DataSource):
        def __read_from():
            for frame in data_source:
                if self.is_closed:
                    break
                self(frame)
        self.read_from_thread = threading.Thread(target=__read_from)
        self.read_from_thread.start()
        if isinstance(data_source, EventDrivenDataSource):
            # print(f"Driving {self} from event driven source {data_source}")
            data_source.drive(self)
        
    def write_to(self, data_sink: DataSink):
        def __write_to():
            while True:
                if self.is_closed:
                    break
                data, is_final = self.msg_queue.get()
                self.msg_queue.task_done()
                if is_final:
                    break
                if data is not None:
                    data_sink(data)
        self.write_to_thread = threading.Thread(target=__write_to)
        self.write_to_thread.start()
            
    def close(self):
        super(DataStreamer, self).close()
        self.is_closed = True
        self.msg_queue.put((None, True))
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

class EventDrivenDataStreamer(DataStreamer, EventDrivenDataSource):
    def __init__(self) -> None:
        super(EventDrivenDataStreamer, self).__init__()
        pass
        
    def write_to(self, data_sink: DataSink):
        super().write_to(data_sink)
        self.drive(data_sink)
