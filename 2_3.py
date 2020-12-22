import numpy as np
import math
import matplotlib.pyplot as plt

class VI:
    def _init_(self,N,mu,lamda,a,b,prec,seed):
        self.seed=seed
        self.N=N
        self.mu=mu
        self.prec=prec
        self.lamda=lamda
        self.a=a
        self.b=b
        self.a_N=[]
        self.b_N=[]
        self.mu_N=[]
        self.lamda_N=[]
    
    def generate_data(self):
        self.data=np.random.normal(self.mu,np.sqrt(self.b/self.a),self.N)
    
    def calculate(self):
        a_0=0
        b_0=0
        mu_0=0
        lamda_0=0
        #calculate alpha_N
        a_N=a_0+self.N/2
        #calculate mu_N
        mu_N=(lamda_0*mu_0+np.sum(self.data))/(lamda_0+self.N)
        #iteratally calculate b_N and lambda_N
        b_N=1
        lamda_N=1
        old_lamda=-1
        iteration=0
        self.a_N.append(a_N)
        self.b_N.append(b_N)
        self.lamda_N.append(lamda_N)
        self.mu_N.append(mu_N)
        while True:
            iteration+=1
            #expectation of mu and mu^2
            old_lamda=lamda_N
            E_mu=mu_N
            E_mu_2=mu_N**2 + 1/lamda_N
            s=sum([self.data[i]**2+E_mu_2-2*mu_N*self.data[i] for i in range(len(self.data))])/2
            b_N=b_0+s+lamda_0*(E_mu+mu_0**2-2*mu_0*mu_N)
            lamda_N=(lamda_0+self.N)*a_N/b_N
            if abs(old_lamda-lamda_N)/lamda_N<0.01:
                break
            self.a_N.append(a_N)
            self.b_N.append(b_N)
            self.lamda_N.append(lamda_N)
            self.mu_N.append(mu_N)

        
    
    def plot(self):
        for i in range(len(self.a_N)):
            numberpoint = 1000
            X, Y = np.meshgrid(np.linspace(self.mu - 0.5, self.mu + 0.5, numberpoint), np.linspace(0.01, 2*self.a/self.b, numberpoint))
            Z = self.gamma_pdf(Y, self.a_N[i], self.b_N[i]) * self.normal_pdf(X, self.mu_N[i], 1/self.lamda_N[i])
            Z_exact = self.gamma_pdf(Y, self.a, self.b) * self.normal_pdf(X, self.mu, 1/(self.lamda*Y)) * self.D_pdf(self.data, X, Y)
            plt.contour(X, Y, Z, levels=5, colors = ["blue"])
            plt.contour(X, Y, Z_exact, levels=5, colors = ["red"])
            plt.title("exact(red)&infer(blue) mu=" + str(self.mu) + ", lambda = " +str(self.lamda) + ", a=" + str(self.a) + ", b=" + str(self.b)+"iter="+str(i))
            plt.xlabel("mu")
            plt.ylabel("tau")
            plt.savefig('size='+str(self.N)+'mu='+str(self.mu)+"lambda="+str(self.lamda) + "a=" + str(self.a) + "b=" + str(self.b)+"iter"+str(i)+".png")
            plt.clf()
    def gamma_pdf(self,X,shape, rate):
        return rate**shape * X**(shape-1) * np.exp(-rate * X) / math.gamma(shape)

    def normal_pdf(self,X,mu, var):
        return 1/np.sqrt(2 * np.pi * var) * np.exp(- ((X - mu)**2) / (2*var))

    def D_pdf(self,D, mu, tau):
        N = len(D)
        var = 0
        for i in range(N):
            var += (D[i] - mu)**2
        return (tau/(2 * np.pi))**(N/2) * np.exp( -tau * var/2)



s=VI()
s._init_(100,0,0.1,1,10,1,223)
s.generate_data()
s.calculate()
s.plot()