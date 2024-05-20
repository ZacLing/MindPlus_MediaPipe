class PIDController:
    def __init__(self, Kp, Ki, Kd, delta_t=1):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.delta_t = delta_t
        self.integral = 0
        self.previous_error = 0

    def compute(self, setpoint, measurement):
        error = setpoint - measurement
        self.integral += error * self.delta_t
        derivative = (error - self.previous_error) / self.delta_t
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        self.previous_error = error
        return output


class MultiDimensionalPIDController(PIDController):
    def __init__(self, Kp, Ki, Kd, num_dimensions=2, delta_t=1):
        super().__init__(Kp, Ki, Kd, delta_t)
        self.num_dimensions = num_dimensions
        self.integrals = [0] * num_dimensions
        self.previous_errors = [0] * num_dimensions

    def compute(self, setpoints, measurements):
        if len(setpoints) != self.num_dimensions or len(measurements) != self.num_dimensions:
            raise ValueError("Setpoints and measurements must match the number of dimensions")

        outputs = []
        for i in range(self.num_dimensions):
            error = setpoints[i] - measurements[i]
            self.integrals[i] += error * self.delta_t
            derivative = (error - self.previous_errors[i]) / self.delta_t
            output = (self.Kp * error) + (self.Ki * self.integrals[i]) + (self.Kd * derivative)
            self.previous_errors[i] = error
            outputs.append(output)
        
        return outputs

class SWG_filter:
    def __init__(self, window_size=3, num_dimensions=2):
        # 初始化窗口大小和维度
        self.window_size = window_size
        self.num_dimensions = num_dimensions
        # 初始化一个长度为window_size, 维度为num_dimensions的队列
        self.queue = [[] for _ in range(num_dimensions)]

    def compute(self, x):
        # 将维度为num_dimensions的向量x入队
        for i in range(self.num_dimensions):
            self.queue[i].append(x[i])
            # 如果队列长度超过window_size, 则移除最早的元素
            if len(self.queue[i]) > self.window_size:
                self.queue[i].pop(0)
        
        # 对每个维度中的队列元素求平均并输出
        averages = []
        for i in range(self.num_dimensions):
            averages.append(sum(self.queue[i]) / len(self.queue[i]))
        
        return averages