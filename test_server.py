import requests,base64,time

INFERENCE_TOKEN="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLWViODI5ZDcxIiwibmFtZSI6IiIsImRlc2NyaXB0aW9uIjoiIiwiZXhwIjoyMDY0ODA3MDc0fQ.nE16bEuS8eZjLQ1Dx3Z-QcQL0-UromcZlsREHuat6fP4yd55tl7N9aGD_84dwEmxsjuf872Rpv0ccFNiPmUdmGbQp4mhW-uVNvBw-VQV3MNcMa5cwHOG-LDSq28A4AxqAxhqZCVrc3npLdKVNZ5etsU0mTr9cK_szKSBwoR4nzgBOKFibKo8ztfM9TuyPlQEq3DdrAGRaxwuRz7OsFdI6xvqSyyzTQ6eoi2tkllbFi8fhl3RnGOn04w1MgfkO8M6M4QiHWOrMz3vpe0Vvx0MIsewEtskTtxM9MHrC22rhRUllnA6HsloSz_oiesiY5HjaeHGUbN0DojOts4shBOAeA"
API_URL="https://api.cortex.cerebrium.ai/v4/p-eb829d71/mtailor"

def test_image(image_file):
    with open(image_file,"rb") as f:
        data=f.read()
    img_b64=base64.b64encode(data).decode("utf8")
    payload={"image":img_b64}
    headers={"Content-Type":"application/json","Authorization":f"Bearer {INFERENCE_TOKEN}"}
    response=requests.post(f"{API_URL}/predict",json=payload,headers=headers)
    if response.status_code==200:
        result=response.json()
        return result["result"]["predicted_class"]
    else:
        print("Error: ",response.status_code)
        return None


def preset_test():
    test_images=["n01440764_tench.jpeg","n01667114_mud_turtle.JPEG"]
    labels=[0,35]
    results=[]
    for i,img in enumerate(test_images):
        pred=test_image(img)
        results.append(pred)
        print(img,": predicted ",pred," expected ",labels[i])
    if results==labels:
        print("Test passed!")
    else:
        print("Test failed!")

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset',action='store_true')
    parser.add_argument('--image',default='n01440764_tench.jpeg')
    args = parser.parse_args()
    start_time=time.time()
    if args.preset:
        preset_test()
    else:
        result=test_image(args.image)
        print("Class ID for ",args.image,": ",result)
    
    print("Time: ",time.time() - start_time,"s")