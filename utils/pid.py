def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

class PID:
    def __init__(self, kp, ki, kd, out_limit=100.0, i_limit=5.0, d_lpf_alpha=0.7, deadband=0.02):
        """
        Initialize a PID controller with various parameters and anti-windup features.
        
        Args:
            kp (float): Proportional gain
            ki (float): Integral gain
            kd (float): Derivative gain
            out_limit (float): Output saturation limit, default 100.0
            i_limit (float): Integral term anti-windup limit, default 5.0
            d_lpf_alpha (float): Derivative low-pass filter smoothing factor (0-1), default 0.7
            deadband (float): Error deadband threshold where integral accumulation stops, default 0.02
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_limit = out_limit      # Maximum output magnitude
        self.i_limit = i_limit          # Maximum integral accumulation
        self.d_lpf_alpha = d_lpf_alpha  # Derivative low-pass filter coefficient

        self.integral = 0.0     # Integral accumulator
        self.prev_err = 0.0     # Previous error for derivative calculation
        self.prev_d = 0.0       # Previous filtered derivative value
        self.initialized = False  # Flag indicating if controller has processed first step
        self.deadband = deadband  # Error threshold for disabling integral

    def reset(self):
        """Reset the controller state (integral, previous values, initialization flag)"""
        self.integral = 0.0
        self.prev_err = 0.0
        self.prev_d = 0.0
        self.initialized = False

    def step(self, err, dt):
        """
        Execute one PID control step.
        
        Args:
            err (float): Current error (setpoint - measurement)
            dt (float): Time step since last update (must be positive)
            
        Returns:
            float: PID control output
        """
        if dt <= 0:
            dt = 1e-3

        # Only accumulate integral when outside deadband to prevent windup
        use_integral = abs(err) > self.deadband

        # Integral: accumulate only when allowed, with anti-windup clamping
        if use_integral:
            self.integral += err * dt
            self.integral = clamp(self.integral, -self.i_limit, self.i_limit)
        else:
            # Gradually reduce integral when in deadband to prevent sticking
            self.integral *= 0.95

        # Derivative: calculate error derivative and apply low-pass filtering
        d_raw = (err - self.prev_err) / dt if self.initialized else 0.0
        d = self.d_lpf_alpha * self.prev_d + (1 - self.d_lpf_alpha) * d_raw

        # PID output calculation
        u = self.kp * err + self.ki * self.integral + self.kd * d
        u = clamp(u, -self.out_limit, self.out_limit)

        # Additional anti-windup: slightly bleed integral when output saturates
        if abs(u) >= self.out_limit - 1e-6:
            self.integral *= 0.99

        # Update state variables
        self.prev_err = err
        self.prev_d = d
        self.initialized = True
        return u
    