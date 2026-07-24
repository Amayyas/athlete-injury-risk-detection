# Deploying the live demo to Hugging Face Spaces

The Space is a thin wrapper over the container image published to `ghcr.io`: it pulls
the image (which already carries a trained model) and runs the dashboard. No rebuild,
no model download — the demo is live in the time it takes HF to pull the image.

You only need to do this once; afterwards the Space tracks the `:latest` release.

### 1. Make the container image public (one click)

The image is published privately by default; HF must be able to pull it anonymously.

- GitHub → your profile → **Packages** → `athlete-injury-risk-detection`
- **Package settings** → **Danger Zone** → **Change visibility** → **Public**

### 2. Create the Space

- <https://huggingface.co/new-space>
- **Owner**: you · **Space name**: `athlete-injury-risk` (or anything)
- **SDK**: **Docker** → **Blank**
- **Hardware**: the free CPU basic tier is enough
- Create

### 3. Add the two files

Copy `deploy/huggingface/Dockerfile` and `deploy/huggingface/README.md` from this repo
into the Space repo (both at its root), then push:

```bash
git clone https://huggingface.co/spaces/<you>/athlete-injury-risk hf-space
cp deploy/huggingface/Dockerfile deploy/huggingface/README.md hf-space/
cd hf-space && git add . && git commit -m "Athlete injury risk demo" && git push
```

HF builds and starts the Space automatically. First boot takes a minute or two while it
pulls the image.

### 4. Send me the URL

Once it's live (`https://huggingface.co/spaces/<you>/athlete-injury-risk`), give me the
link and I'll wire the **live-demo badge** into the top of the main README.

---

**If the pull fails**, the image tag is the usual culprit: `:latest` exists only after a
versioned release. Pin `deploy/huggingface/Dockerfile` to a tag you know exists (e.g.
`:v0.2.0`) and push again.
