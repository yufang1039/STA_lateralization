from pathlib import Path
import pandas as pd
import numpy as np
import numpyro
import jax.numpy as jnp
import jax
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
import numpyro.distributions as dist
from numpyro.infer import Predictive
import arviz as az
import matplotlib.pyplot as plt
from typing import Callable, Mapping, Any, Optional, Tuple, Dict, Union
from numpyro.optim import Adam
from numpyro.distributions import constraints
from jax.scipy.special import ndtr, log_ndtr
from numpyro.distributions.util import promote_shapes
from scipy.stats import gaussian_kde

ModelSpec = Mapping[str, Any]  # key: "func" (Callable) + any kwarg

class RTModel:
    """A class for analyzing reaction time data with hierarchical models."""
    
    def __init__(self, dataname, phase, group, 
                 csv_path: Optional[str] = None, model_name: str = 'model',
                 prior: Optional[Dict] = None):
        """
        Initialize RTModel.
        
        Args:
            phase: Phase number or list of phase numbers
            group: Group number or list of group numbers
            csv_path: Optional path to CSV file to load data
            model_name: Name of the model to use (default: 'model')
            prior: Optional dictionary of prior parameters
        """
        self.df = None
        if csv_path:
            self.load_data(csv_path)
        
        # Get model function by name
        model_func = getattr(self, model_name)
        self.model = self.make_model_spec(model_func)
        self.dataname = dataname
        self.phase = phase
        self.group = group
        self.get_logRT_norm()
        self.data, self.subj_idx, self.repi_idx, self.subjects = self.build_dataset(self.dataname, self.phase, self.group)


    def render_model(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)):
        """Render the model structure using a closure-compatible call to numpyro.render_model."""
        
        model_func = self.model["func"]
        n_subjects = len(self.subjects)
        
        # Prepare closure
        def model_closure():
            return model_func(
                self.data,
                self.repi_idx,
                n_subjects)
        
        # Render the model structure
        fig = numpyro.render_model(
            model_closure,
            render_distributions=True,
            model_args=None,
            model_kwargs=None
        )
        
        # Save if path is provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model structure saved to {save_path}")
        
        return fig
    
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Read a CSV into a DataFrame and ensure required columns exist."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        self.df = pd.read_csv(csv_path)

        required = {"subj", "phase", "repi", "group", "RT"}
        if not required.issubset(set(self.df.columns)):
            missing = ", ".join(required - set(self.df.columns))
            raise ValueError(f"Input CSV missing required columns: {missing}")
        return self.df

    def get_logRT_norm(self):
        """
        Return z-scored log-RT for each trial,
        where the normalization (μ, σ) is computed
        within each subject × phase block.
    """

        if "logRT_norm" not in self.df.columns:
            # 1. log
            log_rt = np.log(self.df["RT"].to_numpy(dtype=float))
            self.df["logRT"] = log_rt                             

            # 2.mean and sd, subject × phase x block
            grp = self.df.groupby(["subj", "phase", "block"])["logRT"]
            mean = grp.transform("mean")
            std  = grp.transform("std", ddof=0).replace(0, np.nan)  # σ=0 时设 NaN

            # 3. z-score；
            self.df["logRT_norm"] = (log_rt - mean) / std
            self.df["logRT_norm"] = self.df["logRT_norm"].fillna(0.0)


        #return self.df["logRT_norm"].values
    
    def get_rt(self, phase: int, repi: int, group, subject=None) -> np.ndarray:
        """Return RT array filtered by phase / repi / group (and optional subject)."""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        mask = (self.df["phase"] == phase) & (self.df["repi"] == repi) & (self.df["group"] == group)
        if subject is not None:
            mask &= self.df["subj"] == subject
        return self.df.loc[mask, "RT"].values
    
    def build_dataset(self, dataname: str, phase: Union[int, list], group: Union[int, list]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract log‑RT, subject index and repi index arrays for modelling.
        
        Args:
            phase: Phase number or list of phase numbers
            group: Group number or list of group numbers
            
        Returns:
            Tuple of (log_rt, subj_idx, repi_idx, subjects)
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Convert inputs to lists
        if isinstance(phase, (int, float)):
            phase = [phase]
        if isinstance(group, (int, float)):
            group = [group]

        # generating synthetic data here
        #     
        # Filter data
        df_pg = self.df[(self.df["phase"].isin(phase)) & (self.df["group"].isin(group))]
        
        if df_pg.empty:
            raise ValueError(f"No rows for phase={phase}, group={group}")

        # Get unique subjects and create mapping
        subjects = df_pg["subj"].unique()
        subj_lookup = {s: i for i, s in enumerate(subjects)}
        
        # Process data using vectorized operations
        data = df_pg[dataname].values
        subj_idx = np.array([subj_lookup[s] for s in df_pg["subj"]], dtype=int)
        repi_idx = df_pg["repi"].values.astype(int) - 1  # 0‑based: 0..5

        return data, subj_idx, repi_idx, subjects
    
    
    @staticmethod
    def make_model_spec(func: Callable, **fixed_kwargs) -> ModelSpec:
        """Create a model specification."""
        return {"func": func, "kwargs": fixed_kwargs}
    
    
    
    @staticmethod
    def model(data, repi_idx, n_subjects, prior=None):
        """Hierarchical change‑point model with learning effect."""
        prior = prior or {}  # ✅ safeguard if prior is None

        # learning effect
        # a = numpyro.sample("a", dist.Normal(0.0,1))        # 初始 z-分数
        # b = numpyro.sample("b", dist.HalfNormal(1))         # 衰减速度 (正),
        # by wen
        a = numpyro.sample("a", prior.get("a", dist.Normal(0.0, 1.0)))
        b = numpyro.sample("b", prior.get("b", dist.HalfNormal(1.0)))

        # state change
        # delta = numpyro.sample("delta", dist.HalfNormal(1))      # z-SD 单位,
        # delta = numpyro.sample("delta", prior.get("delta", dist.HalfNormal(1.0)))

        mu_t = numpyro.sample("mu_t", dist.Uniform(0, 5)) # 0-5
        # sig_t = numpyro.sample("sig_t", dist.HalfCauchy(0.5)) # 0.5, 0.1 , narrower for tstar
        sig_t = numpyro.sample("sig_t", prior.get("sig_t", dist.HalfCauchy(0.5)))

        rep_vals = jnp.arange(6)  # 0..5 inclusive
        log_prob = -((rep_vals - mu_t) ** 2) / (2.0 * sig_t**2)
        probs = jnp.exp(log_prob - jnp.max(log_prob))
        probs = probs / jnp.sum(probs)  # 使用 jnp.sum 而不是 .sum()
        t_star = numpyro.sample("t_star", dist.Categorical(probs=probs))  # 0‑based
        
        # jump = jnp.where(repi_idx < t_star+1, 0.0, -delta)
        mu_learn = jnp.where(repi_idx < t_star+1, a - b * repi_idx, a - b * t_star)
       
        # variance of RT
        # sigma = numpyro.sample("sigma", dist.HalfCauchy(1.0))
        sigma = numpyro.sample("sigma", prior.get("sigma", dist.HalfCauchy(1.0)))

        # Per-repi_idx standard deviation (length = 6)
        # Expand sigma to 6 dimensions, each with its own prior
        # with numpyro.plate("repi", 6):
        #     sigma_raw = numpyro.sample("sigma", dist.HalfCauchy(0.5))  # shape (6,)
        # sigma = sigma_raw[repi_idx]  # align with actual obs indices

        # Likelihood
        # mu_obs = mu_learn + jump
        mu_obs = mu_learn

        if data is None:
            n_obs = repi_idx.shape[0]          
        else:
            n_obs = data.shape[0]


        with numpyro.plate("obs", n_obs):
            numpyro.sample("rt_obs",
                           dist.Normal(mu_obs, sigma),
                           obs=data)

    def run_mcmc(
        self,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        num_chains: int = 1,
        chain_method: str = "parallel",
        rng_key: int = 0,
        prior: Optional[Dict] = None,
    ) -> MCMC:
        """
        Run MCMC inference using NUTS.
        
        Args:
            num_warmup: Number of warmup steps
            num_samples: Number of samples
            num_chains: Number of chains
            chain_method: Chain method ("parallel" or "sequential")
            rng_key: Random key for reproducibility
            prior: Optional dictionary of prior parameters
            
        Returns:
            MCMC object
        """
        model_func = self.model["func"]
        
        # Use instance variables
        data = self.data
        repi_idx = self.repi_idx
        n_subjects = len(self.subjects)
        
        kernel = NUTS(model_func)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, 
                   num_chains=num_chains, chain_method=chain_method)
        
        # Run MCMC with correct parameter order
        mcmc.run(jax.random.PRNGKey(rng_key), data, repi_idx, n_subjects, prior=prior)
        return mcmc
    
    def get_idata(
        self,
        inference_result: Union[MCMC, Any],
    ) -> az.InferenceData:
        """Get ArviZ inference data object from MCMC or SVI results."""
        model_func = self.model["func"]
        fixed_kw = self.model.get("kwargs", {})
        
       
        posterior_samples = inference_result.get_samples()
       
        # Handle t_star for two-state model
        if "mu_t" in posterior_samples:
            posterior_samples["t_star"] = jnp.floor(posterior_samples["mu_t"]).astype("int32")

        rng_key = jax.random.PRNGKey(0)

        predictive = Predictive(
            model_func,
            posterior_samples=posterior_samples,
            return_sites=["log_rt_obs"],
        )
        
        # Use instance variables
        log_rt = self.log_rt
        subj_idx = self.subj_idx
        repi_idx = self.repi_idx
        n_subjects = len(self.subjects)
        
        # Prepare arguments for predictive
        args = [rng_key, None, subj_idx]
        if repi_idx is not None:
            args.append(repi_idx)
        if n_subjects is not None:
            args.append(n_subjects)
        args.extend([v for v in fixed_kw.values()])
        
        ppc_samples = predictive(*args)

        if isinstance(inference_result, MCMC):
            # Use from_numpyro for MCMC results
            idata = az.from_numpyro(
                posterior=inference_result,
                posterior_predictive=ppc_samples,
            )
        else:
            # Use from_dict for SVI results
            idata = az.from_dict(
                posterior=posterior_samples,
                posterior_predictive={"log_rt_obs": ppc_samples["log_rt_obs"]},
                coords={"chain": [0], "draw": [0]}
            )

    def get_mode_kde(samples, bandwidth=None):
        samples = np.asarray(samples)

        kde = gaussian_kde(samples, bw_method=bandwidth)
        x_grid = np.linspace(samples.min(), samples.max(), 1000)
        density = kde(x_grid)

        mode_value = x_grid[np.argmax(density)]
        return mode_value
