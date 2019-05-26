import { createStyles } from '@material-ui/core';

const webStyle = ({palette, spacing, breakpoints})=>createStyles({
    mainSign:{
        backgroundColor: palette.primary.main,
        height:'100%',
      },
      bsGen:{
        width:'100%',
        color: '#FFFFFF',
        'font-weight': 'bold',
        'font-size': '4em'
      },
      alignCenter:{
        width:'90%',
        marginLeft:'5%',
      },
      inpField: {
        width: '20em'
      },
      main: {
        width: 'auto',
        marginLeft: spacing.unit * 20,
        marginRight: spacing.unit * 20,
        paddingTop: spacing.unit * 20,
        paddingBottom: spacing.unit * 2,
        '& > div': {
          marginTop: spacing.unit * 8,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          paddingTop: spacing.unit * 4,
          paddingBottom: spacing.unit * 4,
          paddingLeft: spacing.unit,
          paddingRight: spacing.unit,
          '& button': {
            width: '100%'
          },
          '& > h1': {
            marginBottom: spacing.unit * 4
          },
          '& > span': {
            textAlign: 'center'
          },
          '& > form': {
            marginBottom: spacing.unit * 2,
            '& > section': {
              marginBottom: spacing.unit * 4,
              '& > div': {
                width: '100%',
                '&:not(last-child)': {
                  marginBottom: spacing.unit
                }
              }
            }
          }
        }
      },
});

export default webStyle;
