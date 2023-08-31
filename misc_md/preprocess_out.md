The final output structure of preprocessing should look like this:
```
./test_images
├── aligned_images
│   └── valid
│       ├── 60922.png
│       ├── 63252.png
│       └── 66646.png
├── face_segment    <--- Face parsing
│   └── valid
│       ├── anno
│       │   ├── anno_60922.png
│       │   ├── anno_63252.png
│       │   └── anno_66646.png
│       └── vis
│           ├── res_60922.png
│           ├── res_63252.png
│           └── res_66646.png
├── images
│   └── valid
│       ├── 60922.jpg
│       ├── 63252.jpg
│       └── 66646.jpg
├── params      <--- Parameters from DECA, Arcface and Diff-AE
│   └── valid
│       ├── ffhq-valid-cam-anno.txt
│       ├── ffhq-valid-exp-anno.txt
│       ├── ffhq-valid-faceemb-anno.txt
│       ├── ffhq-valid-light-anno.txt
│       ├── ffhq-valid-pose-anno.txt
│       ├── ffhq-valid-shadow-anno.txt
│       └── ffhq-valid-shape-anno.txt
└── rendered_images     <--- Rendered images from DECA
    ├── deca_masked_face_wclip
    │   └── valid
    │       ├── 60922.png
    │       ├── 63252.png
    │       └── 66646.png
    └── deca_masked_face_woclip
        └── valid
            ├── 60922.npy
            ├── 63252.npy
            └── 66646.npy
```