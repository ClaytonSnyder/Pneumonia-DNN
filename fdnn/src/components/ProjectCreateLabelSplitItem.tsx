import { TextField } from "@mui/material";

interface ProjectCreateLabelSplitItemProps {
    identifier: string;
    split: number;
    setLabel: (key: string, value: number) => void;
}

export function ProjectCreateLabelSplitItem(props: ProjectCreateLabelSplitItemProps) {
    const setSplit = (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        props.setLabel(props.identifier, Number(event.target.value as string));
    };

    return (
        <TextField
            fullWidth
            style={{ marginTop: 15 }}
            label={props.identifier}
            variant="outlined"
            value={props.split}
            onChange={setSplit}
        />
    );
}
