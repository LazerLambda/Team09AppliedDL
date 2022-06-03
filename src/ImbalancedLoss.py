class ImbalancedLoss:

    @staticmethod
    def sum_exp(x: torch.Tensor, sgn: int):
        return torch.sum(
            sigmoid_loss(
                output=x,
                label=(sgn * torch.ones(x.shape[0]))))
    
    @staticmethod
    def sigmoid_loss(output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute Sigmoid Loss Function.
        
        Compute:
        :param output: predicted value.

        :return: loss
        """
        product: torch.Tensor = torch.mul(output, label)
        denum: torch.Tensor = torch.add(1, torch.exp(product))
        return torch.div(1, denum)

    def imbalanced_nnpu(
            self,
            pred_p: torch.Tensor,
            pred_u: torch.Tensor,
            p: float,
            p_: float) -> torch.Tensor:
        """Compute ImbalancednnPU-Loss

        Implementation according to https://www.ijcai.org/proceedings/2021/0412.pdf
        
        :param pred_p: Positive labeled data.
        :param pred_u: Unlabeled data.
        :param p: Proportion of labeled to unlabeled data.
        :param p_: Proportion of upsampled data.

        :returns: Loss for batch. 
        """
        
        nu: int = pred_u.shape[0]
        cu: float = (1 - p_) / (nu * (1 - p))

        np: int = pred_p.shape[0]
        cp: float = (1 - p_) * p / (np * (1 - p))

        return cu * self.sum_exp(pred_u, -1) - cp * self.sum_exp(pred_p, -1)

    def nn_balancePN(
            self,
            pred_p: torch.Tensor,
            pred_u: torch.Tensor,
            p: float,
            p_: float) -> torch.Tensor:
        """Compute nnBalancePN.

        Implementation according to https://arxiv.org/abs/1703.00593.

        :param pred_p: Positive labeled data.
        :param pred_u: Unlabeled data.
        :param p: Proportion of labeled to unlabeled data.
        :param p_: Proportion of upsampled data.

        :returns: Loss for batch. 
        """
        np: int = pred_p.shape[0]
        clipped_imbnnPU: torch.Tensor = torch.max(
            torch.Tensor([0]),
            self.imbalanced_nnpu(
                pred_p=pred_p,
                pred_u=pred_u,
                p=p,
                p_=p_
            ))
        return p_ / np * self.sum_exp(pred_p, 1) + clipped_imbnnPU

        
    
