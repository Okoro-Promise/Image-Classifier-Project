import argparse


def get_input_args():
    """
    Retrieves and parses the  command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these command line arguments. If 
    the user fails to provide some or all of the arguments, then the default 
    values are used for the missing arguments. 
   
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create command line arguments as mentioned above using add_argument() from ArgumentParser method
    parser.add_argument('data_dir', action="store", metavar='data_dir', default="./flowers/")
    parser.add_argument('--save_dir', action="store", default="./checkpoint.pth")
    parser.add_argument('--arch', action="store", default="vgg16", choices=['vgg16', 'resnet18'])
    parser.add_argument('--learning_rate', action="store", type=float,default=0.001)
    parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default= 300)
    parser.add_argument('--epochs', action="store", default= 3, type=int)
    parser.add_argument('--dropout', action="store", type=float, default=0.5)
    parser.add_argument('--gpu', action="store", default="gpu")
    
    # Return Parser
    return parser.parse_args()



def get_test_inputs():
    """
    Retrieves and parses the  command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these command line arguments. If 
    the user fails to provide some or all of the arguments, then the default 
    values are used for the missing arguments. 
   
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image_path', metavar='image_path', type=str, default= 'flowers/test/100/image_07902.jpg')
    parser.add_argument('checkpoint', metavar='checkpoint', type=str, default='checkpoint.pth')
    parser.add_argument('--top_k', action='store', dest="top_k", type=int, default=5)
    parser.add_argument('--category_names', action='store', dest='category_names', type=str, default='cat_to_name.json')
    parser.add_argument('--gpu', action='store', default= "gpu")
    
    # Return Parser
    return parser.parse_args()