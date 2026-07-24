# Changelog

## [0.2.0](https://github.com/Amayyas/athlete-injury-risk-detection/compare/injury-risk-v0.1.0...injury-risk-v0.2.0) (2026-07-24)


### Features

* **api:** FastAPI service over the shared inference seam ([d743565](https://github.com/Amayyas/athlete-injury-risk-detection/commit/d743565b35935eabe4b5a05369bb9a0901369591))
* **api:** FastAPI service over the shared inference seam ([d5aa72e](https://github.com/Amayyas/athlete-injury-risk-detection/commit/d5aa72e9f3bac8103b6862656be646e03dcb2065)), closes [#21](https://github.com/Amayyas/athlete-injury-risk-detection/issues/21)
* **api:** HTTP client returning the same types as the local predictor ([52fe568](https://github.com/Amayyas/athlete-injury-risk-detection/commit/52fe5686a31e771c18488660d8318a9b37b2b837))
* automated releases (SemVer) with model artifacts ([3797543](https://github.com/Amayyas/athlete-injury-risk-detection/commit/3797543ef36153bed72ba8ff69bd9314d6845b25))
* **cli:** bind the dashboard and API to all interfaces ([bdafc4e](https://github.com/Amayyas/athlete-injury-risk-detection/commit/bdafc4eaa9b77f34358fd02bfdd5bab700ac1770))
* **cli:** injury-risk serve, plus the api extra and locks ([3319482](https://github.com/Amayyas/athlete-injury-risk-detection/commit/331948211394287db5ca7e33be1d5d8c7ccfccc0))
* **cli:** one entry point for the whole pipeline ([28859a0](https://github.com/Amayyas/athlete-injury-risk-detection/commit/28859a05ceecb4d3ed657a0a3c10115695064f5c)), closes [#9](https://github.com/Amayyas/athlete-injury-risk-detection/issues/9)
* **dashboard:** consume the API when one is configured ([60ed131](https://github.com/Amayyas/athlete-injury-risk-detection/commit/60ed131e93a20ab41a7cd16270e05ffefcbc2c9e)), closes [#22](https://github.com/Amayyas/athlete-injury-risk-detection/issues/22)
* **dashboard:** consume the API, with a local fallback ([12ff869](https://github.com/Amayyas/athlete-injury-risk-detection/commit/12ff869dbcb169fae8e922cd77658b3257c671bb))
* **dashboard:** Streamlit app with real-time risk scoring ([64b4d6f](https://github.com/Amayyas/athlete-injury-risk-detection/commit/64b4d6f1c180001c79020a99e8fa5b2146d1dbd5))
* **dashboard:** use the trained model, live, next to the rule score ([a2ec082](https://github.com/Amayyas/athlete-injury-risk-detection/commit/a2ec08223e39991f5b736c5eb6e1e14d5bb5c87b)), closes [#18](https://github.com/Amayyas/athlete-injury-risk-detection/issues/18) [#19](https://github.com/Amayyas/athlete-injury-risk-detection/issues/19)
* **dashboard:** use the trained model, live, with per-athlete SHAP ([06f70f8](https://github.com/Amayyas/athlete-injury-risk-detection/commit/06f70f8b0f447ec2464581d735dedc1e0cc5666e))
* **data:** Kaggle download, SIRP-600 loader and synthetic generator ([6cb3b58](https://github.com/Amayyas/athlete-injury-risk-detection/commit/6cb3b58e1aa67db621da2d0fb1ab050753e544dc))
* **features:** ACWR, rolling features and composite risk score ([ab4daa3](https://github.com/Amayyas/athlete-injury-risk-detection/commit/ab4daa3e0cd13cece928fbf2d968ba5fcb9669f6))
* **inference:** a single serving seam for the model ([2d547a5](https://github.com/Amayyas/athlete-injury-risk-detection/commit/2d547a5a2e4a6991ef75a4841ceb1d35d48de1a8))
* **infra:** container image + publish to ghcr.io ([4cf0936](https://github.com/Amayyas/athlete-injury-risk-detection/commit/4cf09362494e7a2a072848cfbeab5ff5779337d1))
* **infra:** container image, with a model baked in ([e0f3f64](https://github.com/Amayyas/athlete-injury-risk-detection/commit/e0f3f64963bbec156f17bf6cce27bd4060499d9e))
* **ml:** a guard on model quality, with thresholds sized from measured variance ([d4485b4](https://github.com/Amayyas/athlete-injury-risk-detection/commit/d4485b4fbf0656b871141554739ccf941e01ebcd)), closes [#24](https://github.com/Amayyas/athlete-injury-risk-detection/issues/24)
* **ml:** cost-based decision threshold ([16b5c75](https://github.com/Amayyas/athlete-injury-risk-detection/commit/16b5c75622cc2b80655f99aa018885fdefb20e34))
* **ml:** forward-looking target and past-only features ([e391754](https://github.com/Amayyas/athlete-injury-risk-detection/commit/e39175495ac5c09049bdafda8d067fd0e895da97)), closes [#13](https://github.com/Amayyas/athlete-injury-risk-detection/issues/13) [#3](https://github.com/Amayyas/athlete-injury-risk-detection/issues/3)
* **ml:** isotonic calibration, evaluated on reliability not ranking ([0380986](https://github.com/Amayyas/athlete-injury-risk-detection/commit/03809866828bd962196ad54868f6f223ee1b7eae)), closes [#16](https://github.com/Amayyas/athlete-injury-risk-detection/issues/16)
* **ml:** PR-AUC as headline metric, with reference points to read it against ([f9fd3c3](https://github.com/Amayyas/athlete-injury-risk-detection/commit/f9fd3c3299dad12d2e60460b7bb1cd93f96bc79e))
* **ml:** replace the circular target with real injury events ([79c7b36](https://github.com/Amayyas/athlete-injury-risk-detection/commit/79c7b36891bc05f30a82d0cefe68ae24e692e219))
* **ml:** simulate real injury events instead of labelling with the rules ([6d1c493](https://github.com/Amayyas/athlete-injury-risk-detection/commit/6d1c493f3a772f0b7f8b3e99678ebed72e186954)), closes [#12](https://github.com/Amayyas/athlete-injury-risk-detection/issues/12)
* **ml:** tune every candidate, not just the favourite ([1674bca](https://github.com/Amayyas/athlete-injury-risk-detection/commit/1674bcabdfeeec107df4e37e231eac82bd535772)), closes [#15](https://github.com/Amayyas/athlete-injury-risk-detection/issues/15)
* **ml:** tuning, calibration and a cost-based threshold ([e7f71c3](https://github.com/Amayyas/athlete-injury-risk-detection/commit/e7f71c382a4280490a39f6713946a565e608d58d))
* **models:** XGBoost + SMOTE training, tuning and baseline benchmark ([0a87c8e](https://github.com/Amayyas/athlete-injury-risk-detection/commit/0a87c8efd08ad6aa2e44e34c9fc31e528f6d6947))
* **release:** fetch the delivered model from a GitHub Release ([c1d5743](https://github.com/Amayyas/athlete-injury-risk-detection/commit/c1d57435da18ae3e3077bebe5106f1f2bd467466))
* resolve where predictions come from, API first with a local fallback ([0983945](https://github.com/Amayyas/athlete-injury-risk-detection/commit/0983945ba2de7ad4ac40fe609e9e2b3d7b520292))
* single-source the version from pyproject ([6130ba2](https://github.com/Amayyas/athlete-injury-risk-detection/commit/6130ba2c1552655d67e20e32318eaa50b673ebbe))
* **viz:** pick the SHAP explainer from the model ([8a76afd](https://github.com/Amayyas/athlete-injury-risk-detection/commit/8a76afd58244b7aa8072993254eb9182eb3209a1))
* **viz:** SHAP explainability (summary + waterfall plots) ([770c091](https://github.com/Amayyas/athlete-injury-risk-detection/commit/770c091c0b9185691fd2a34305ba9f9c32640704))


### Bug fixes

* **build:** stop the lock from leaking a local path ([30cd59d](https://github.com/Amayyas/athlete-injury-risk-detection/commit/30cd59dc40b4429fb8d429ad0a5afe9264defe2b))
* **ml:** athlete-level CV leakage + SMOTE/scaler ordering ([089d2c8](https://github.com/Amayyas/athlete-injury-risk-detection/commit/089d2c8419f03a6e3dd7eaf4eda422cbce13e256))
* **ml:** athlete-level CV leakage and SMOTE/scaler ordering ([c758155](https://github.com/Amayyas/athlete-injury-risk-detection/commit/c758155b74c630504a111ddc3a35aa2d31730925)), closes [#1](https://github.com/Amayyas/athlete-injury-risk-detection/issues/1) [#2](https://github.com/Amayyas/athlete-injury-risk-detection/issues/2)
* **notebooks:** repair the EDA notebook broken by the ML redesign ([7a72c12](https://github.com/Amayyas/athlete-injury-risk-detection/commit/7a72c12d3e81ccd5bc8eb36054922503fb8645f0))


### Refactoring

* central config + public dataset API ([0267cb5](https://github.com/Amayyas/athlete-injury-risk-detection/commit/0267cb518e6615c60788d1bd04aa122da20d35fe))
* central config and public dataset API ([6c15054](https://github.com/Amayyas/athlete-injury-risk-detection/commit/6c150547c16c798797881ec76aa2c6ef41e2b830)), closes [#4](https://github.com/Amayyas/athlete-injury-risk-detection/issues/4) [#5](https://github.com/Amayyas/athlete-injury-risk-detection/issues/5)
* **features:** make the factor list the score's decomposition ([751dced](https://github.com/Amayyas/athlete-injury-risk-detection/commit/751dced498cb80a54a6730b70b8a1aa3f8e2d5b8)), closes [#6](https://github.com/Amayyas/athlete-injury-risk-detection/issues/6)
* **models:** define the candidates once ([873550d](https://github.com/Amayyas/athlete-injury-risk-detection/commit/873550dddd35ae66b2b9f2eee04bfba012bc41bb))
* rule scoring as a tested module + broadened test suite ([50b5fcd](https://github.com/Amayyas/athlete-injury-risk-detection/commit/50b5fcd18fa610f48522c68b4c52cc7911cd0c54))


### CI/CD

* add quality workflow (lint, format, types, tests, coverage) ([10a1e2c](https://github.com/Amayyas/athlete-injury-risk-detection/commit/10a1e2cabd13c6fbfa29c48e424f7b24bd901ba3)), closes [#23](https://github.com/Amayyas/athlete-injury-risk-detection/issues/23)
* align release.yml action versions with the rest of the repo ([f59a9b1](https://github.com/Amayyas/athlete-injury-risk-detection/commit/f59a9b11552b27a38765a501b24f672462c22356))
* automated releases (release-please) that carry the model ([b60e91e](https://github.com/Amayyas/athlete-injury-risk-detection/commit/b60e91e7feb135d80517a1f973fcce73699f5bbf)), closes [#28](https://github.com/Amayyas/athlete-injury-risk-detection/issues/28) [#29](https://github.com/Amayyas/athlete-injury-risk-detection/issues/29)
* build and publish the image to ghcr.io ([0768755](https://github.com/Amayyas/athlete-injury-risk-detection/commit/076875596898c9084e2e636e02119e5ebd833377)), closes [#30](https://github.com/Amayyas/athlete-injury-risk-detection/issues/30)
* bump actions to remove the Node 20 deprecation warning ([31c3e77](https://github.com/Amayyas/athlete-injury-risk-detection/commit/31c3e776faa03b997a8b69d86a37faf26d6d8b7e))
* execute notebooks, scan for secrets and audit dependencies ([33fd800](https://github.com/Amayyas/athlete-injury-risk-detection/commit/33fd8005d1bed20e2361972b5f799af5e4dc2669))
* guard model quality, not just code ([7a4dcde](https://github.com/Amayyas/athlete-injury-risk-detection/commit/7a4dcdebe18e3262d627900abb1e988409a69948))
* notebooks in CI, security scanning, pre-commit ([bc947ba](https://github.com/Amayyas/athlete-injury-risk-detection/commit/bc947ba4868f7a91f38a8cc9a824ec96c9f05039))
* quality workflow (lint, format, types, tests, coverage) ([a581cbd](https://github.com/Amayyas/athlete-injury-risk-detection/commit/a581cbdf3fa5ba193dcd294b8711465a92fe74b7))
* run the model-quality guard on every pull request ([ebd886f](https://github.com/Amayyas/athlete-injury-risk-detection/commit/ebd886fce1a032dbfb86a279f0ff0a09f1abd2c7))


### Build & dependencies

* **deps:** Bump actions/checkout from 5 to 7 ([6634097](https://github.com/Amayyas/athlete-injury-risk-detection/commit/663409786c38b1fb7d4c1a6353f1a1b7c5a6d952))
* **deps:** Bump actions/checkout from 5 to 7 ([fbf0087](https://github.com/Amayyas/athlete-injury-risk-detection/commit/fbf0087f8e59d96183c302c15ef78d90c1adb748))
* **deps:** Bump actions/setup-python from 6 to 7 ([f1841f8](https://github.com/Amayyas/athlete-injury-risk-detection/commit/f1841f88202f09e5f1e41f51991666ac22b67764))
* **deps:** Bump actions/setup-python from 6 to 7 ([7501433](https://github.com/Amayyas/athlete-injury-risk-detection/commit/750143330204ae3192c1fde8eb6c2db9010c7759))
* **deps:** Bump actions/upload-artifact from 5 to 7 ([7313bc8](https://github.com/Amayyas/athlete-injury-risk-detection/commit/7313bc8103c91cd9761c4473bae3efcc2121d3fc))
* **deps:** Bump actions/upload-artifact from 5 to 7 ([97d5b42](https://github.com/Amayyas/athlete-injury-risk-detection/commit/97d5b422b5aaab19c019f06b62d242a7b52e6790))
* **deps:** Bump gitleaks/gitleaks-action from 2 to 3 ([0291c51](https://github.com/Amayyas/athlete-injury-risk-detection/commit/0291c5106f7c13242a4bd5358936f0707ae424cf))
* **deps:** Bump gitleaks/gitleaks-action from 2 to 3 ([1f98001](https://github.com/Amayyas/athlete-injury-risk-detection/commit/1f98001282e195369d5f069eb1236e31939c3dfc))
* **deps:** Bump the github-actions group with 5 updates ([f359fe6](https://github.com/Amayyas/athlete-injury-risk-detection/commit/f359fe63f8eb85d23ce4eebe02d9c5e7975c5a2b))
* **deps:** Bump the github-actions group with 5 updates ([2ad9399](https://github.com/Amayyas/athlete-injury-risk-detection/commit/2ad93993d05a49dbbceb687e479785f908d1a27c))
* Makefile as the single description of how to run the project ([5fea613](https://github.com/Amayyas/athlete-injury-risk-detection/commit/5fea613790e4cef39dfd0f16a5c0c4ff0aa86acc))
* pin dependencies with lock files ([1e961a8](https://github.com/Amayyas/athlete-injury-risk-detection/commit/1e961a8f2e477e0f5b5ab608138dc7df37b89559)), closes [#11](https://github.com/Amayyas/athlete-injury-risk-detection/issues/11)


### Documentation

* describe the two readings the dashboard now shows ([d7f6757](https://github.com/Amayyas/athlete-injury-risk-detection/commit/d7f6757eac078ce8c5d8d568b20e331d206f2d84))
* document notebook execution, pre-commit and security scanning ([fe7a2e1](https://github.com/Amayyas/athlete-injury-risk-detection/commit/fe7a2e101265583af6f0d35e79aac2154c6f3e57)), closes [#25](https://github.com/Amayyas/athlete-injury-risk-detection/issues/25)
* document running the dashboard against the API ([b9b8302](https://github.com/Amayyas/athlete-injury-risk-detection/commit/b9b83023dbc840f6131ae7befa4aaf6699fabd8f))
* document the CLI, the Makefile and the reproducible installs ([1ab3689](https://github.com/Amayyas/athlete-injury-risk-detection/commit/1ab3689b3932e0ea3977737bdb9aa46a52a56bc4))
* document the REST API ([c5fafa6](https://github.com/Amayyas/athlete-injury-risk-detection/commit/c5fafa61498fd23c5020b78b3c695ae11701052a))
* document the risk-factor decomposition and the new layout ([2bb9ccd](https://github.com/Amayyas/athlete-injury-risk-detection/commit/2bb9ccd1909a713dd17da6d8542af2210df8a3c6))
* document versioning, releases and model artifacts ([dd4da95](https://github.com/Amayyas/athlete-injury-risk-detection/commit/dd4da9570ac2ffc7abe9c8534a3dc6d26b4ac509))
* explain what the model guard catches, and what it does not ([78718ba](https://github.com/Amayyas/athlete-injury-risk-detection/commit/78718ba07c031a57918f5dc7312c9da997add74f))
* README, EDA notebook, dashboard screenshot and result reports ([f6102dc](https://github.com/Amayyas/athlete-injury-risk-detection/commit/f6102dc892481b86843d2f014dc4b5c6e8c05d81))
* refresh the dashboard screenshot with the model prediction ([f55c7ed](https://github.com/Amayyas/athlete-injury-risk-detection/commit/f55c7ed038c26a6a34956eb7093a56ed1f24e323))
* refresh the dashboard screenshot with the model prediction ([b26653d](https://github.com/Amayyas/athlete-injury-risk-detection/commit/b26653dead8ba8c837dd290bd1d7d2ee9b1676db))
* report the model selection, the calibration trade-off and the ACWR result ([56dfc2f](https://github.com/Amayyas/athlete-injury-risk-detection/commit/56dfc2f10d9b6a92975b0b390feb080176ee1ece))
* rewrite the results around the new predictive task ([77dda51](https://github.com/Amayyas/athlete-injury-risk-detection/commit/77dda51d6559fea6be1e6e901c9fa41591e6bd92))
