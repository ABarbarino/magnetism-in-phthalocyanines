import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


############ FUNÇÕES ##########
def inicializar_rede(L):
    '''
    Inicializa uma rede quadrada LxL com spins aleatórios ±1 
    Parâmetros:
        L: tamanho da rede.
    Retorna:
        rede: configuração inicial dos spins na rede.
    '''
    return np.random.choice([-1, 1], size=(L, L))

####
def energia_total(rede, J):
    '''
    Calcula a energia total do sistema de Ising bidimensional com interações de primeiros vizinhos.

    Parametros:
        rede: configuração atual dos spins;
        J: acoplamento dos íons metálicos.
    Retona:
        energia: energia total.
    '''                                                     
    L = rede.shape[0]
    energia = 0
    for i in range(L):
        for j in range(L):
            S = rede[i, j]
            vizinhos = (
                rede[(i + 1) % L, j] +
                rede[i, (j + 1) % L] +
                rede[(i - 1) % L, j] +
                rede[i, (j - 1) % L]
            )
            energia += -J * S * vizinhos       
    return energia / 2  # Cada par de spins é contado duas vezes, então dividimos por 2
    
###
def magnetizacao_total(rede):
    '''
    Magnetização total do sistema definida como a soma de todas as configurações de spin.

    ParÂmetros: 
        rede: configuração atual dos spins.
    Retona:
        M: magnetização total.
    '''
    
    return np.sum(rede)

###
def passo_metro(rede, T, J):
    '''
    Executa um passo de Monte Carlo com o algoritmo de Metropolis.

    Parâmetros:
        rede: ;
        T: temperatura;
        J: acoplamento dos íons metálicos.
    Retorna:
        rede: configuração de spins atualizada da rede.
    '''
    L = rede.shape[0]

    for _ in range(L*L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        S = rede[i, j]
        vizinhos = (
            rede[(i + 1) % L, j] +
            rede[i, (j + 1) % L] +
            rede[(i - 1) % L, j] +
            rede[i, (j - 1) % L]
        )
        dE = 2 * J * S * vizinhos       # variação de energia                     
        if dE < 0 or np.random.rand() < np.exp(-dE / T):
            rede[i, j] *= -1
    return rede

###
def simular(L, temps, equil_steps, prod_steps,J):
    '''
    Executa a simulação de Monte Carlo para um conjunto de temperaturas.
    Calculamos a energia média, magnetização média, calor específico,
    susceptibilidade magnética e estimamos a temperatura crítica.

    Parâmetros:
        L: tamanho da rede;
        temps: lista das temperaturas;
        equil_steps: número de passos de equilibração;
        prod_steps: número de passosde produção;
        J: acoplamento dos íons metálicos.
    Retorna:
        energia_m: lista da energia média por spin;
        mag_m: lista da magnetização média por spin;
        c_v: lista do calor específico;
        susc: lista da susceptibilidade magnética;
        Tc_média: estimativa da temperatura crítica.
        '''
    energia_m = []
    mag_m = []
    c_v = []
    susc = []

    for T in tqdm(temps):
        rede = inicializar_rede(L)

        # Equilibração
        for _ in range(equil_steps):
            passo_metro(rede, T, J)

        energias = []
        magnetizacoes = []

        # Produção
        for _ in range(prod_steps):
            passo_metro(rede, T, J)
            
            e = energia_total(rede, J)
            m = magnetizacao_total(rede)
            
            energias.append(e)
            magnetizacoes.append(abs(m))

        e_m = np.mean(energias)/(L*L)
        m_m = np.mean(magnetizacoes)/(L*L)
        c = np.var(energias)/(T**2 * L*L)
        x = np.var(magnetizacoes)/(T * L*L)

        energia_m.append(e_m)
        mag_m.append(m_m)
        c_v.append(c)
        susc.append(x)

    # Estimativa da temperatura crítica pela média entra os picos do calor específico e da susceptibilidade
    Tc_cv = temps[np.argmax(c_v)]
    Tc_susc = temps[np.argmax(susc)]
    Tc_media = (Tc_cv + Tc_susc) / 2


    print(f"Temperatura crítica estimada (pico de C_v): {Tc_cv:.3f}")
    print(f"Temperatura crítica estimada (pico de χ): {Tc_susc:.3f}")
    print(f"Temperatura crítica média estimada: {Tc_media:.3f}")

    return energia_m, mag_m, c_v, susc, Tc_media

###

def plotar_todos_resultados(L, Eaf, Efm, nome_do_arquivo):
    '''
    Executa a simulação do modelo de Ising 2D e gera automaticamente todos os gráficos termodinâmicos de interesse.

    O parâmetro de acoplamento efetivo J é calculado a partir das energias ferromagnética (Efm) e antiferromagnética (Eaf).

    Parâmetros:
        L: Tamanho da rede;
        Eaf: Energia do estado antiferromagnético;
        Efm: Energia do estado ferromagnético;
        nome_do_arquivo: string para identificar o caso.

    Retorna:
        Tc: Temperatura críticaestimada.
    '''
    J = (Eaf - Efm)/ 2
    temps = np.linspace(0.001, 1, 40)
    energia, magnet, calor_especifico, susc_magn, Tc = simular(L=L,temps=temps, equil_steps=500, prod_steps=500,J=J)

    # Energia
    plt.figure(figsize=(10,6))
    plt.plot(temps, energia, '-o')
    plt.title('Energia Média por Spin', fontsize=20)
    plt.xlabel('Temperatura ($k_B T / J$)', fontsize=15)
    plt.ylabel('Energia ($E/J$)', fontsize=15)
    plt.grid(True)
    plt.savefig(f"{nome_do_arquivo}_energia.png", dpi=300)
    plt.show()

    # Magnetização
    plt.figure(figsize=(10,6))
    plt.plot(temps, magnet, 'o-')
    plt.title('Magnetização Média por Spin', fontsize=20)
    plt.xlabel("Temperatura ($k_B T$)", fontsize=15)
    plt.ylabel("Magnetização ($|M|$ por spin)", fontsize=15)
    plt.grid(True)
    plt.savefig(f"{nome_do_arquivo}_magnetizacao.png", dpi=300)
    plt.show()

    # Calor específico
    plt.figure(figsize=(10,6))
    plt.plot(temps, calor_especifico, '-o')
    plt.axvline(Tc, linestyle='--', alpha=0.7)
    plt.text(0.95, 0.95, f"Tc = {Tc:.2f}", color='royalblue', transform=plt.gca().transAxes,ha='right', va='top', fontsize=12)
    plt.title('Calor Específico', fontsize=20)
    plt.xlabel('Temperatura ($k_B T / J$)', fontsize=15)
    plt.ylabel('$C_V / k_B$', fontsize=15)
    plt.grid(True)
    plt.savefig(f"{nome_do_arquivo}_calor_especifico.png", dpi=300)
    plt.show()

    # Susceptibilidade
    plt.figure(figsize=(10,6))
    plt.plot(temps, susc_magn, '-o')
    plt.axvline(Tc, linestyle='--', alpha=0.7)
    plt.text(0.95, 0.95, f"Tc = {Tc:.2f}", color='royalblue', transform=plt.gca().transAxes, ha='right', va='top', fontsize=12)
    plt.title('Susceptibilidade Magnética', fontsize=20)
    plt.xlabel('Temperatura ($k_B T$)', fontsize=15)
    plt.ylabel('$\\chi / (1/J)$', fontsize=15)
    plt.grid(True)
    plt.savefig(f"{nome_do_arquivo}_susceptibilidade.png", dpi=300)
    plt.show()

    return Tc



