import numpy as np

class Cartpole3D:
    def __init__(self):
        self.g = 9.8
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.half_len = 0.5
        self.total_mass = self.pole_mass + self.cart_mass
        self.force_mag = 1.0
        self.dt = 0.02 #euler int

        self.x = 0.0
        self.x_dot = 0.0
        self.y = 0.0
        self.y_dot = 0.0
        self.alpha = 0.0
        self.alpha_dot = 0.0
        self.beta = 0.0
        self.beta_dot = 0.0

        self.x_bound = 9
        self.y_bound = 9
        self.angle_bound = 1.0 

        self.max_steps = 1000
        self.step_count = 0
    
    def reset(self):
        self.x = 0.0
        self.x_dot = 0.0
        self.y = 0.0
        self.y_dot = 0.0
        self.alpha = np.random.uniform(-1, 1)
        self.alpha_dot = 0.0
        self.beta = np.random.uniform(-0.1, 0.1)
        self.beta_dot = 0.0
        self.step_count = 0
        return self._get_state()
    

    def _get_state(self):
        return np.array([
            self.x, self.x_dot,
            self.y, self.y_dot,
            self.alpha, self.alpha_dot,
            self.beta, self.beta_dot
        ])

    def time_step(self, action):
        force_x = np.clip(action[0], -self.force_mag, self.force_mag)
        force_y = np.clip(action[1], -self.force_mag, self.force_mag)
        
        M = self.total_mass
        m = self.pole_mass
        l = self.half_len
        I = (1/3) * m * l**2
        a = self.alpha
        b = self.beta
        a_dot = self.alpha_dot
        b_dot = self.beta_dot
        
        ca, sa = np.cos(a), np.sin(a)
        cb, sb = np.cos(b), np.sin(b)
        
        # Mass matrix (4×4)
        M_mat = np.array([
            [M+m, 0,    m*l*ca*cb,      -m*l*sa*sb],
            [0,   M+m,  m*l*ca*sb,       m*l*sa*cb],
            [m*l*ca*cb, m*l*ca*sb,  I+m*l**2,     0],
            [-m*l*sa*sb, m*l*sa*cb, 0, (I+m*l**2)*sa**2]
        ])

        M_mat += np.eye(4) * 1e-6
        
        # RHS
        rhs = np.array([
            force_x - m*l*(sa*cb*a_dot**2 + 2*ca*sb*a_dot*b_dot + sa*cb*b_dot**2),
            force_y - m*l*(sa*sb*a_dot**2 - 2*ca*cb*a_dot*b_dot + sa*sb*b_dot**2),
            -m*self.g*l*sa + m*l*sa*ca*b_dot**2,
            -2*(I+m*l**2)*sa*ca*a_dot*b_dot
        ])
        
        # Solve
        acc = np.linalg.solve(M_mat, rhs)
        x_acc, y_acc, alpha_acc, beta_acc = acc
        
        # Euler integration
        self.x_dot += x_acc * self.dt
        self.y_dot += y_acc * self.dt
        self.alpha_dot += alpha_acc * self.dt
        self.beta_dot += beta_acc * self.dt
        
        self.alpha_dot = np.clip(self.alpha_dot, -10.0, 10.0)
        self.beta_dot = np.clip(self.beta_dot, -10.0, 10.0)
        self.x_dot = np.clip(self.x_dot, -10.0, 10.0)
        self.y_dot = np.clip(self.y_dot, -10.0, 10.0)

        self.x += self.x_dot * self.dt
        self.y += self.y_dot * self.dt
        self.alpha += self.alpha_dot * self.dt
        self.beta += self.beta_dot * self.dt
        
        self.step_count += 1
    
        done = (abs(self.x) > self.x_bound or 
                abs(self.y) > self.y_bound or 
                abs(self.alpha) > self.angle_bound or
                self.step_count >= self.max_steps)  # Time limit
        
        reward = 1.0 if not done else 0.0
        return self._get_state(), reward, done
