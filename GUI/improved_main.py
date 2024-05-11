import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
from tkinter.font import nametofont
import tkinter as tk
from PIL import ImageTk, Image
import numpy as np
from roboticstoolbox import SerialLink, RevoluteDH, PrismaticDH
import matplotlib
import matplotlib.pyplot as plt
import sympy as syp
from sympy.printing import latex
from io import BytesIO as StringIO
matplotlib.use('TkAgg')
class RoboticProgram(ttkb.Window):
    
    def __init__(self):
        super().__init__()

        self.title("Kinematic Analysis")
        self.geometry('1280x720')
        self.resizable(width=False, height=False)
        ML = ttkb.Labelframe(bootstyle='light')
        ML.place(x = 75, y = 150, relwidth=0.3, relheight=1)
        PL = ttkb.Canvas()
        PL.place(x = 512, y = 0, relwidth=0.7, relheight=1)
        self.windowTitle = ttkb.Label(ML, text="Kinematic Analysis Calculator", font = 'Helvetica 18 bold')
        self.windowTitle.pack(pady=20)
        
        palletizingRobot = ImageTk.PhotoImage(Image.open('robot.jpg').resize((1280, 720)))
        img_robot = tk.Label(PL, image=palletizingRobot)
        img_robot.dontloseit = palletizingRobot
        img_robot.pack(fill="both")
        
        FKin = ttkb.Button(ML, text = "Forward Kinematics and Jacobian", command = FkinWindow, bootstyle="primary")
        FKin.pack(pady=20, padx = 10)
        
        IKin = ttkb.Button(ML, text = "Inverse Kinematics", command= IkinWindow, bootstyle="primary")
        IKin.pack(pady=20, padx = 10)
        
        
        
class Window(RoboticProgram):
    def __init__(self):
        self.windowTitle = ttkb.Toplevel(master = robot)
        self.windowTitle.geometry("1000x500")
        LL = ttkb.Labelframe(master=self.windowTitle, text = "Link Length and Joint Variables")
        LL.place(x=0, y=0, width=200, height=200)
        LL.columnconfigure((0,1,2), weight=1, uniform="a")
        LL.rowconfigure((0,1,2), weight=1, uniform="a")
        a1L = ttkb.Label(LL, text = ("a1 = "))
        self.a1data = ttkb.Entry(LL,width=5)
        cm1 = ttkb.Label(LL, text = ("cm"))
        
        a2L = ttkb.Label(LL, text = ("a2 = "))
        self.a2data = ttkb.Entry(LL,width=5)
        cm2 = ttkb.Label(LL, text = ("cm"))
        
        a3L = ttkb.Label(LL, text = ("a3 = "))
        self.a3data = ttkb.Entry(LL,width=5)
        cm3 = ttkb.Label(LL, text = ("cm"))
        
        a1L.grid(row=0,column=0, padx=5, pady=5)
        self.a1data.grid(row=0,column=1, padx=5, pady=5)
        cm1.grid(row=0,column=2, padx=5, pady=5)
        
        a2L.grid(row=1,column=0, padx=5, pady=5)
        self.a2data.grid(row=1,column=1, padx=5, pady=5)
        cm2.grid(row=1,column=2, padx=5, pady=5)
        
        a3L.grid(row=2,column=0, padx=5, pady=5)
        self.a3data.grid(row=2,column=1, padx=5, pady=5)
        cm3.grid(row=2,column=2, padx=5, pady=5)
        
        JV= ttkb.Labelframe(master=self.windowTitle, text = "Joint Variables")
        JV.place(x=300, y=0, width=200, height=200)
        JV.columnconfigure((0,1,2), weight=1, uniform="a")
        JV.rowconfigure((0,1,2), weight=1, uniform="a")
        t1L = ttkb.Label(JV, text = ("T1 = "))
        self.T1data = ttkb.Entry(JV,width=5)
        deg1 = ttkb.Label(JV, text = ("deg"))
        
        t2L = ttkb.Label(JV, text = ("T2 = "))
        self.T2data = ttkb.Entry(JV,width=5)
        deg2 = ttkb.Label(JV, text = ("deg"))
        
        d3L = ttkb.Label(JV, text = ("d3 = "))
        self.d3data = ttkb.Entry(JV,width=5)
        cm4 = ttkb.Label(JV, text = ("cm"))
        
        t1L.grid(row=0, column=0, padx=5, pady=5)
        self.T1data.grid(row=0,column=1,  padx=5, pady=5)
        deg1.grid(row=0, column=2, padx=5, pady=5)
        
        t2L.grid(row=1, column=0, padx=5, pady=5)
        self.T2data.grid(row=1, column=1, padx=5, pady=5)
        deg2.grid(row=1, column=2, padx=5, pady=5)
        
        d3L.grid(row=2, column=0, padx=5, pady=5)
        self.d3data.grid(row=2, column=1, padx=5, pady=5)
        cm4.grid(row=2, column=2, padx=5, pady=5)
        
        PV = ttkb.Labelframe(master=self.windowTitle, text = "Position Vector")
        PV.place(x=150, y=350, width=200, height=200)
        PV.columnconfigure((0,1,2), weight=1, uniform="a")
        PV.rowconfigure((0,1,2), weight=1, uniform="a")
        XL = ttkb.Label(PV, text = ("X = "))
        self.Xdata = ttkb.Entry(PV,width=5)
        cm5 = ttkb.Label(PV, text = ("cm"))
        
        YL = ttkb.Label(PV, text = ("Y = "))
        self.Ydata = ttkb.Entry(PV,width=5)
        cm6 = ttkb.Label(PV, text = ("cm"))
        
        ZL = ttkb.Label(PV, text = ("Z = "))
        self.Zdata = ttkb.Entry(PV,width=5)
        cm7 = ttkb.Label(PV, text = ("cm"))
        
        XL.grid(row=0, column=0, padx=5, pady=5)
        self.Xdata.grid(row=0, column=1, padx=5, pady=5)
        cm5.grid(row=0, column=2, padx=5, pady=5)
        
        YL.grid(row=1, column=0, padx=5, pady=5)
        self.Ydata.grid(row=1, column=1, padx=5, pady=5)
        cm6.grid(row=1, column=2, padx=5, pady=5)
        
        ZL.grid(row=2, column=0, padx=5, pady=5)
        self.Zdata.grid(row=2, column=1, padx=5, pady=5)
        cm7.grid(row=2, column=2, padx=5, pady=5)
        
    def reset(self):        
        self.a1data.delete(0,'end')
        self.a2data.delete(0, 'end')
        self.a3data.delete(0, 'end')
        self.T1data.delete(0, 'end')
        self.T2data.delete(0,'end')
        self.d3data.delete(0, 'end')
        self.Xdata.delete(0, 'end')
        self.Ydata.delete(0, 'end')
        self.Zdata.delete(0, 'end')
        
class FkinWindow(Window):	
    def __init__(self):
        super().__init__()
        
        self.windowTitle.geometry("750x600")
        self.Xdata.config(state= ttkb.DISABLED)
        self.Ydata.config(state= ttkb.DISABLED)
        self.Zdata.config(state= ttkb.DISABLED)

        self.windowTitle.title("Foward Kinematics")
        
        
        BF = ttkb.Labelframe(master=self.windowTitle)
        BF.place(x=125, y=250, width=250, height = 75)
        BF.columnconfigure((0,1,2), weight=1, uniform="a")
        BF.rowconfigure((0), weight=1, uniform="a")
        forward = ttkb.Button(BF, text = "Foward", command=self.fkin, bootstyle="primary-outline")
        forward.grid(row=0, column=0)
        reset = ttkb.Button(BF, text = "Reset", command=self.reset, bootstyle="secondary-outline")
        reset.grid(row=0, column=1)
        htm = ttkb.Button(BF, text = "Table", command=self.showTable, bootstyle="secondary")
        htm.grid(row=0, column=2)
                
        JVV = ttkb.LabelFrame(master=self.windowTitle, text = "Joint Variable Velocities")
        JVV.place(x=500, y=0, width=250, height=200)
        JVV.columnconfigure((0,1,2), weight=1, uniform="a")
        JVV.rowconfigure((0,1,2), weight=1, uniform="a")
        T1_dot= ttkb.Label(JVV, text = ("Theta_1 = "))
        self.T1_dotdata = ttkb.Entry(JVV,width=5)
        cms1 = ttkb.Label(JVV, text = ("cm/s"))
        
        T2_dot= ttkb.Label(JVV, text = ("Theta_1 = "))
        self.T2_dotdata = ttkb.Entry(JVV,width=5)
        cms2= ttkb.Label(JVV, text = ("cm/s"))
        
        d3_dot= ttkb.Label(JVV, text = ("Theta_1 = "))
        self.d3_dotdata = ttkb.Entry(JVV,width=5)
        cms3 = ttkb.Label(JVV, text = ("cm/s"))
        
        EEV = ttkb.LabelFrame(master=self.windowTitle, text = "End Effector Velocities")
        EEV.place(x=500, y=225, width=250, height=300)
        EEV.columnconfigure((0,1,2), weight=1, uniform="a")
        EEV.rowconfigure((0,1,2), weight=1, uniform="a")
        
        x_dot= ttkb.Label(EEV, text = ("x_dot= "))
        self.x_dotdata = ttkb.Entry(EEV,width=5)
        cms4 = ttkb.Label(EEV, text = ("cm/s"))
        
        y_dot= ttkb.Label(EEV, text = ("y_dot = "))
        self.y_dotdata = ttkb.Entry(EEV,width=5)
        cms5 = ttkb.Label(EEV, text = ("cm/s"))
        
        z_dot= ttkb.Label(EEV, text = ("z_dot = "))
        self.z_dotdata = ttkb.Entry(EEV,width=5)
        cms6 = ttkb.Label(EEV, text = ("cm/s"))
        
        T_x_dot= ttkb.Label(EEV, text = ("T_x_dot= "))
        self.T_x_dotdata = ttkb.Entry(EEV,width=5)
        cms7 = ttkb.Label(EEV, text = ("cm/s"))
        
        T_y_dot= ttkb.Label(EEV, text = ("T_y_dot = "))
        self.T_y_dotdata = ttkb.Entry(EEV,width=5)
        cms8 = ttkb.Label(EEV, text = ("cm/s"))
        
        T_z_dot= ttkb.Label(EEV, text = ("T_z_dot = "))
        self.T_z_dotdata = ttkb.Entry(EEV,width=5)
        cms9 = ttkb.Label(EEV, text = ("cm/s"))
        
        T1_dot.grid(row = 0, column = 0, padx=5, pady=5)
        self.T1_dotdata.grid(row = 0, column = 1, padx=5, pady=5)
        cms1.grid(row=0, column=2, padx=5, pady=5)
        
        T2_dot.grid(row = 1, column = 0, padx=5, pady=5)
        self.T2_dotdata.grid(row = 1, column = 1, padx=5, pady=5)
        cms2.grid(row=1, column=2, padx=5, pady=5)
        
        d3_dot.grid(row = 2, column = 0, padx=5, pady=5)
        self.d3_dotdata.grid(row = 2, column = 1, padx=5, pady=5)
        cms3.grid(row=2, column=2, padx=5, pady=5)
        
        x_dot.grid(row = 0, column = 0, padx=5, pady=5)
        self.x_dotdata.grid(row = 0, column = 1, padx=5, pady=5)
        cms4.grid(row = 0, column = 2, padx=5, pady=5)
        
        y_dot.grid(row = 1, column = 0, padx=5, pady=5)
        self.y_dotdata.grid(row = 1, column = 1, padx=5, pady=5)
        cms5.grid(row = 1, column = 2, padx=5, pady=5)
        
        z_dot.grid(row = 2, column = 0, padx=5, pady=5)
        self.z_dotdata.grid(row = 2, column = 1, padx=5, pady=5)
        cms6.grid(row = 2, column = 2, padx=5, pady=5)
        
        T_x_dot.grid(row = 3, column = 0, padx=5, pady=5)
        self.T_x_dotdata.grid(row = 3, column = 1, padx=5, pady=5)
        cms7.grid(row = 3, column = 2, padx=5, pady=5)
        
        T_y_dot.grid(row = 4, column = 0, padx=5, pady=5)
        self.T_y_dotdata.grid(row = 4, column = 1, padx=5, pady=5)
        cms8.grid(row = 4, column = 2, padx=5, pady=5)
        
        T_z_dot.grid(row = 5, column = 0, padx=5, pady=5)
        self.T_z_dotdata.grid(row = 5, column = 1, padx=5, pady=5)
        cms9.grid(row = 5, column = 2, padx=5, pady=5)
        
        update = ttkb.Button(master=self.windowTitle, text="Update", command=self.jacobian)
        update.place(x=575, y=550)
        
        self.robotTB(1,0.5,0.5,0,0,0)
          
    def fkin(self):
        self.Xdata.config(state= ttkb.NORMAL)
        self.Ydata.config(state= ttkb.NORMAL)
        self.Zdata.config(state= ttkb.NORMAL)
        self.Xdata.delete(0, 'end')
        self.Ydata.delete(0, 'end')
        self.Zdata.delete(0, 'end')
        try:
            a1 = float(self.a1data.get()) / 100
            a2 = float(self.a2data.get()) / 100
            a3 = float(self.a3data.get()) / 100
            t1 = float(self.T1data.get()) * (np.pi/180)
            t2 = float(self.T2data.get()) * (np.pi/180)
            d3 = float(self.d3data.get()) / 100
        except ValueError:
            pop_up = ttkb.Toplevel(master= robot)
            label = ttkb.Label(pop_up, text = "Use the approriate syntax (float)")
            label.pack()
            
        parametric_table = [[t1, np.pi/2, 0, a1],
                            [np.pi/2 + t2, np.pi/2, 0, 0 ],
                            [0,0,0,a2+a3+d3]]
        htm = {}
        
        for i in range(3):
            htm[i] = self.dhMatrix(parametric_table[i][0], parametric_table[i][1], parametric_table[i][2], parametric_table[i][3])
            
        for i,j in htm.items():
            print(f"HTM # {i+1}")
            print(np.round(j, 2))
        
        self.H0_1 = htm[0]
        self.H1_2 = htm[1]
        self.H0_2 = np.dot(htm[0], htm[1])
        self.H2_3 = htm[2]
        self.H0_3 = np.dot(np.dot(htm[0], htm[1]), htm[2])
        
        result = np.dot(np.dot(htm[0], htm[1]), htm[2])
        print(np.round(result,2))
        
        if (t1 < -np.pi/2 or t1 > np.pi/2) or (t2 < -np.pi/2 or t2 > np.pi/2) or (d3 < 0 or d3 >=  0.5):
            another_pop_up = ttkb.Toplevel(master=robot)
            qlim_error = ttkb.Label(another_pop_up, text = "The Calculated Joint Limits exceeded the Acutal Joint Limits")
            qlim_error.pack()
        else:
            
            self.Xdata.insert(ttkb.END, np.round(result[0,3] * 100,2))
            self.Ydata.insert(ttkb.END, np.round(result[1,3] * 100,2))
            self.Zdata.insert(ttkb.END, np.round(result[2,3] * 100,2))
            self.Xdata.config(state= ttkb.DISABLED)
            self.Ydata.config(state= ttkb.DISABLED)
            self.Zdata.config(state= ttkb.DISABLED)
            self.robotTB(a1, a2, a3, t1, t2, d3)

    def jacobian(self):
        
        self.x_dotdata.config(state= ttkb.NORMAL)
        self.y_dotdata.config(state= ttkb.NORMAL)
        self.z_dotdata.config(state= ttkb.NORMAL)
        self.T_x_dotdata.config(state= ttkb.NORMAL)
        self.T_y_dotdata.config(state= ttkb.NORMAL)
        self.T_z_dotdata.config(state= ttkb.NORMAL)
        self.x_dotdata.delete(0, 'end')
        self.y_dotdata.delete(0, 'end')
        self.z_dotdata.delete(0, 'end')
        self.T_x_dotdata.delete(0, 'end')
        self.T_y_dotdata.delete(0, 'end')
        self.T_z_dotdata.delete(0, 'end')
        
        R0_0 = np.identity(3)
        z0_1 = np.array([[0],[0],[1]])
        d0_3 = self.H0_3[0:3, 3]
        d0_0 = 0
        
        R0_1 = self.H0_1[0:3, 0:3]
        d0_3 = self.H0_3[0:3, 3]
        d0_1 = self.H0_1[0:3, 3]
        
        R0_2 = self.H0_2[0:3, 0:3]
        z0_0 = np.zeros((3,1))
        
        Jv_1 = np.cross(np.dot(R0_0, z0_1), (d0_3-d0_0), axis = 0)
        Jw_1 = np.dot(R0_0, z0_1)


        Jv_2 = np.cross(np.dot(R0_1, z0_1), (d0_3-d0_1), axis = 0)
        Jw_2 = np.dot(R0_1, z0_1)

        Jv_3 = np.dot(R0_2, z0_1)
        Jw_3 = z0_0

        Jv = np.concatenate([Jv_1, Jv_2, Jv_3], 1)

        Jw = np.concatenate([Jw_1, Jw_2, Jw_3], 1)

        J = np.concatenate([Jv, Jw], 0)
        t = syp.Symbol("t")

        x = syp.Function("x")
        y=  syp.Function("y")
        z = syp.Function("z")
        theta_x = syp.Function("theta_x")
        theta_y = syp.Function("theta_y")
        theta_z = syp.Function("theta_z")

        theta_1 = syp.Function("theta_1")
        theta_2 = syp.Function("theta_2")
        d_3 = syp.Function("d_3")


        joint_variables = syp.Matrix([[syp.diff(theta_1(t))], [syp.diff(theta_2(t))], [syp.diff(d_3(t))]])
        position_variables = syp.Matrix([[syp.diff(x(t))], [syp.diff(y(t))], [syp.diff(z(t))], [syp.diff(theta_x(t))], [syp.diff(theta_y(t))], [syp.diff(theta_z(t))]])
        syp.init_printing()
        print(J)
        Jacobian_matrix = syp.Eq(position_variables, J*joint_variables)


        print(syp.pretty(Jacobian_matrix))
        J_s = np.linalg.det(Jv)

        invJv = np.linalg.inv(Jv)
        
        print(J)
        
        print(J_s)
        
        JV_dot = np.array([[float(self.T1_dotdata.get())], [float(self.T2_dotdata.get())], [float(self.d3_dotdata.get())]])
        
        print(JV_dot)
        
        EV_dot = np.dot(J, JV_dot)
        
        print(EV_dot)
        
        self.x_dotdata.insert(ttkb.END, EV_dot[0, 0])
        self.y_dotdata.insert(ttkb.END, EV_dot[1, 0])
        self.z_dotdata.insert(ttkb.END, EV_dot[2, 0])
        self.T_x_dotdata.insert(ttkb.END, EV_dot[3, 0])
        self.T_y_dotdata.insert(ttkb.END, EV_dot[4, 0])
        self.T_z_dotdata.insert(ttkb.END, EV_dot[5, 0])
        self.x_dotdata.config(state= ttkb.DISABLED)
        self.y_dotdata.config(state= ttkb.DISABLED)
        self.z_dotdata.config(state= ttkb.DISABLED)
        self.T_x_dotdata.config(state= ttkb.DISABLED)
        self.T_y_dotdata.config(state= ttkb.DISABLED)
        self.T_z_dotdata.config(state= ttkb.DISABLED)
        
        singularity_and_jacobian = ttkb.Toplevel()
        singularity_and_jacobian.title("Jacobian Differential Equations and its Jacobian")
        singularity_and_jacobian.geometry("500x500")
        
        singulariity_result = f"The Singularity of the Jacobian is {J_s}"
        singularity = ttkb.Label(singularity_and_jacobian, text= singulariity_result)
        singularity.pack()
        
        def render_latex(formula, fontsize=12, dpi=300, format_='svg'):
            """Renders LaTeX formula into image.
            """
            fig = plt.figure(figsize=(0.01, 0.01))
            fig.text(0, 0, f'${formula}$', fontsize=fontsize)
            buffer_ = StringIO()
            fig.savefig(buffer_, dpi=dpi, transparent=True, format=format_, bbox_inches='tight', pad_inches=0.0)
            plt.close(fig)
            return buffer_.getvalue()

        if __name__ == '__main__':
            image_bytes = render_latex(latex(Jacobian_matrix),fontsize=10, dpi=200, format_='png')
            with open('jacobian_de.png', 'wb') as image_file:
                image_file.write(image_bytes)
        
    def dhMatrix(self, theta, alpha, radius, distance):
        return np.matrix([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), radius*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), radius*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), distance],
            [0,0,0,1]
        ])
    def robotTB(self, a1, a2, a3, t1, t2, d3):
        
        H01 = RevoluteDH(a1, 0, np.pi/2, 0, [-np.pi/2, np.pi/2])
        H12 = RevoluteDH(0, 0, np.pi/2, np.pi/2, [-np.pi/2, np.pi/2])
        H23 = PrismaticDH(0,0,0,a2+a3,[0, 0.5])
        
        sphericalManipulator = SerialLink([H01, H12, H23])
        sphericalManipulator.plot([t1, t2, d3])

    def showTable(self):
        showCanvas = ttkb.Toplevel(self.windowTitle)
        showCanvas.title("Denavit Hartenberg Computation of Spherical Manipulator")
        dhTable = ttkb.Canvas(master=showCanvas)
        dhTable.pack()

        denavithartenberg = ImageTk.PhotoImage(Image.open('Fkin.png').resize((500,500)))
        img_robot = tk.Label(dhTable, image=denavithartenberg)
        img_robot.dontloseit = denavithartenberg
        img_robot.pack(fill="both")

class IkinWindow(Window):
    def __init__(self):
        super().__init__()
        self.windowTitle.geometry("500x500")
        self.T1data.config(state= ttkb.DISABLED)
        self.T2data.config(state= ttkb.DISABLED)
        self.d3data.config(state= ttkb.DISABLED)
        
        self.windowTitle.title("Inverse Kinematics")
        
        BF = ttkb.Labelframe(master=self.windowTitle)
        BF.place(x=125, y=250, relwidth=0.5, relheight=0.15)
        BF.columnconfigure((0,1,2), weight=1, uniform="a")
        BF.rowconfigure((0), weight=1, uniform="a")
        inverse= ttkb.Button(BF, text = "Inverse", command=self.ikin, bootstyle="primary-outline")
        inverse.grid(row=0, column=0)
        reset = ttkb.Button(BF, text = "Reset", command=self.reset, bootstyle="secondary-outline")
        reset.grid(row=0, column=1)
        soln = ttkb.Button(BF, text = "Solution", command=self.showSoln, bootstyle="secondary")
        soln.grid(row=0, column=2)
        
    def ikin(self):
        self.T1data.config(state= ttkb.NORMAL)
        self.T2data.config(state= ttkb.NORMAL)
        self.d3data.config(state= ttkb.NORMAL)
        self.T1data.delete(0, 'end')
        self.T2data.delete(0,'end')
        self.d3data.delete(0, 'end')
        
        try:
            a1 = float(self.a1data.get()) / 100
            a2 = float(self.a2data.get()) / 100
            a3 = float(self.a3data.get()) / 100
            x = float(self.Xdata.get()) / 100
            y = float(self.Ydata.get()) / 100
            z = float(self.Zdata.get()) / 100
        except ValueError:
            pop_up = ttkb.Toplevel(master= robot)
            label = ttkb.Label(pop_up, text = "Use the approriate syntax (float)")
            label.pack()
        try:
            t1, t2, d3 = self.invKins(a1, a2, a3, x, y, z)
        except ZeroDivisionError:
            pop_up = ttkb.Toplevel(master= robot)
            label = ttkb.Label(pop_up, text = "The Inverse Kinematic Calculation is undefined")
            label.pack()
            
        self.T1data.insert(ttkb.END, t1)
        self.T2data.insert(ttkb.END, t2)
        self.d3data.insert(ttkb.END, d3)
        self.T1data.config(state= ttkb.DISABLED)
        self.T2data.config(state= ttkb.DISABLED)
        self.d3data.config(state= ttkb.DISABLED)
            
    def invKins(self, a1, a2, a3, x_03, y_03, z_03):
        s = z_03 - a1
        r = np.sqrt((x_03**2) + (y_03**2))
        theta1 = np.arctan(y_03/x_03) * 180/np.pi
        theta2 = np.arctan(s/r) * 180/np.pi
        d3 = (np.sqrt((r**2) + (s**2)) - a2 - a3) * 100
        return theta1, theta2, d3
    
    def showSoln(self):
        showCanvas = ttkb.Toplevel(self.windowTitle)
        showCanvas.title("Inverse Kinematics Graphical Solution")
        dhTable = ttkb.Canvas(master=showCanvas)
        dhTable.pack()

        denavithartenberg = ImageTk.PhotoImage(Image.open('Ikin.png').resize((500,500)))
        img_robot = tk.Label(dhTable, image=denavithartenberg)
        img_robot.dontloseit = denavithartenberg
        img_robot.pack(fill="both")

class JBinWindow():
    def __init__(self):
        self.windowTitle = tk.Toplevel(master = robot)
        underConstruction = ttkb.Label(self.windowTitle, text = "Hi! This Function is under construction and will be updated once Jacobian matrix is being discussed")
        underConstruction.pack()
robot = RoboticProgram()
robot.style.theme_use('simplex')

default_font = nametofont('TkDefaultFont')
default_font.configure(family="Helvetica", size=12)
robot.mainloop()


