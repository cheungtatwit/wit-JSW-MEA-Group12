import pathlib
import cv2

input_directory = 'Enter input folder here'
# input example: 'C:\\Users\\cheungt\\Desktop\\myJoint\\raw_image'
output_directory = 'Enter output folder here'
# output example: 'C:\\Users\\cheungt\\Desktop\\myJoint\\input'
extension = '.png'


def main():
    global input_directory, output_directory
    input_directory = pathlib.Path(input_directory)
    output_directory = pathlib.Path(output_directory)

    files = list(input_directory.glob(f'*{extension}'))

    for file in files:
        image = cv2.imread(str(file), 0)
        image = cv2.resize(image, (448,448))
        result = crophist_func(image)
        output_path = output_directory/file.name
        print(output_path)
        cv2.imwrite(str(output_path), result)

def crophist_func(image):
    image_crop = image[10:170, 10:170]
    hist = cv2.equalizeHist(image)
    return hist
    

if __name__ == '__main__':
    main()
