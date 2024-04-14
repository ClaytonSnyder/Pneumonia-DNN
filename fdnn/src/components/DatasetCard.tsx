import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Typography from "@mui/material/Typography";
import { IDataset } from "./DatasetPage";
import { Link } from "@mui/material";

export interface IDatasetCardProps {
    dataset: IDataset;
}

export default function DatasetCard(props: IDatasetCardProps) {
    return (
        <Card sx={{ width: 500, marginRight: 2 }}>
            <CardContent>
                <Typography gutterBottom variant="h5" component="div">
                    {props.dataset.name}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                    {props.dataset.description}
                </Typography>
            </CardContent>
            <CardContent style={{ paddingBottom: 5, paddingTop: 5 }}>
                <Typography align="left" variant="button" component="div">
                    Total Images
                </Typography>
                <Typography align="left" variant="overline" component="div">
                    {props.dataset.total_image_count}
                </Typography>
            </CardContent>
            <CardContent style={{ paddingBottom: 5, paddingTop: 5 }}>
                <Typography align="left" variant="button" component="div">
                    Labels
                </Typography>
                <Typography align="left" variant="overline" component="div">
                    {props.dataset.labels.join()}
                </Typography>
            </CardContent>
            <CardContent style={{ paddingBottom: 5, paddingTop: 5 }}>
                <Typography align="left" variant="button" component="div">
                    Source Data
                </Typography>
                {props.dataset.kaggle_datasets_included.map((kaggle) => (
                    <div key={kaggle.identifier} style={{ textAlign: "left" }}>
                        <Link align="left" variant="overline" underline="none" href={kaggle.url}>
                            {kaggle.identifier}
                        </Link>
                    </div>
                ))}
            </CardContent>
        </Card>
    );
}
