from Prior.model import Prior
if __name__ == '__main__':

    prior = Prior(use_cuda=False)

    prior.save('priors/RandomPrior')